from networkx.classes.function import selfloop_edges
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import dgl
import dgl.nn as dglnn
from transformers.models.marian.modeling_marian import MarianSinusoidalPositionalEmbedding as SinusoidalPositionalEmbedding
from transformers.models.bart.modeling_bart import (
    shift_tokens_right,
    BartConfig,
    BartEncoder,
    BartEncoderLayer,
    BartPretrainedModel,
    _expand_mask, _make_causal_mask,
    BartLearnedPositionalEmbedding, 
    BartAttention,
    BartDecoder,
    BartDecoderLayer,
)
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput, Seq2SeqModelOutput,BaseModelOutputWithPastAndCrossAttentions
from utils.CONSTANT import DISCOURSE_RELATIONS

from transformers import BartConfig


name_2_activation_fn_mapping = {
    'tanh':F.tanh,
    'relu':F.relu,
}

class GraphTransformerConfig(BartConfig):

    def __init__(
        self,
        backbone_model ='../pretrained_model/bart_large',        
        model_type='graphtransformer',
        # all_bart_base config
        gt_activation_dropout = 0.1 ,
        gt_activation_function = 'gelu' ,
        gt_add_bias_logits = False ,
        gt_add_final_layer_norm = False ,
        gt_attention_dropout = 0.1 ,
        gt_d_model = 768 ,
        gt_decoder_attention_heads = 12 ,
        gt_decoder_ffn_dim = 3072 ,
        gt_decoder_layerdrop = 0.0 ,
        gt_dropout = 0.1 ,
        gt_encoder_attention_heads = 12 ,
        gt_encoder_ffn_dim = 3072 ,
        gt_encoder_layerdrop = 0.0 ,
        gt_encoder_layers = 6 ,
        gt_init_std = 0.02 ,
        gt_is_encoder_decoder = True ,
        gt_normalize_before = False ,
        gt_normalize_embedding = True ,
        gt_scale_embedding = False,
        conv_activation_fn = 'relu',
        num_beams = 5,
        rezero = 1,
        max_length = 100,
        min_length = 5,
        utt_pooling = 'average',
        gt_pos_embed = '',
        **kwargs,
    ):
        for k,v in kwargs.items():
            if not hasattr(self,k):
                setattr(self,k,v)
        
        pretrained_model_config = BartConfig.from_pretrained(backbone_model)
        for k,v in vars(pretrained_model_config).items():
            if not hasattr(self,k):
                setattr(self,k,v)
        self.gt_pos_embed = gt_pos_embed
        self.conv_activation_fn = conv_activation_fn
        self.utt_pooling = utt_pooling
        self.backbone_model =backbone_model
        self.model_type=model_type
        self.gt_activation_dropout =gt_activation_dropout
        self.gt_activation_function =gt_activation_function
        self.gt_add_bias_logits =gt_add_bias_logits
        self.gt_add_final_layer_norm =gt_add_final_layer_norm
        self.gt_attention_dropout =gt_attention_dropout
        self.gt_d_model =gt_d_model
        self.gt_decoder_attention_heads =gt_decoder_attention_heads
        self.gt_decoder_ffn_dim =gt_decoder_ffn_dim
        self.gt_decoder_layerdrop =gt_decoder_layerdrop
        
        self.gt_dropout =gt_dropout
        self.gt_encoder_attention_heads =gt_encoder_attention_heads
        self.gt_encoder_ffn_dim =gt_encoder_ffn_dim
        self.gt_encoder_layerdrop =gt_encoder_layerdrop
        self.gt_encoder_layers =gt_encoder_layers
        self.gt_init_std =gt_init_std
        self.gt_is_encoder_decoder =gt_is_encoder_decoder
        self.min_length = min_length
        self.gt_normalize_before = gt_normalize_before
        self.gt_normalize_embedding =gt_normalize_embedding
        self.gt_scale_embedding =gt_scale_embedding
        self.num_beams = num_beams
        self.max_length = max_length
        self.rezero = rezero

class GraphTransformerMultiHeadAttentionLayer(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        
        self.config = config
        self.hid_dim = config.d_model
        self.n_heads = config.gt_encoder_attention_heads
        
        assert self.hid_dim % self.n_heads == 0
        self.head_dim = self.hid_dim // self.n_heads
        
        self.fc_q = nn.Linear(self.hid_dim,self.hid_dim)
        self.fc_k = nn.Linear(self.hid_dim,self.hid_dim)
        self.fc_v = nn.Linear(self.hid_dim,self.hid_dim)
        
        self.fc_o = nn.Linear(self.hid_dim,self.hid_dim)
        self.dropout = nn.Dropout(config.gt_attention_dropout)
        self.scale = self.head_dim ** 0.5
        
        self.feature_maps_num = len(config.feature_types)
        self.feature_conv = nn.Sequential(*[nn.Conv2d(
                        in_channels=self.feature_maps_num,
                        out_channels=1,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    ) for _ in range(self.n_heads)])
        # self.combine_conv = nn.Sequential(
        #     *[nn.Conv2d(
        #         in_channels=2,
        #         out_channels=1,
        #         kernel_size=1,
        #         stride=1,
        #         padding=0,
        #     ) for _ in range(self.n_heads)]
        #)
        self.combine_conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv_activation_fn = name_2_activation_fn_mapping[config.conv_activation_fn]
        
        # self.p_dyna_linear = nn.Sequential(
        #                                     *[nn.Linear(self.head_dim,1) for _ in range(self.n_heads)]
        #                                   )
        if 'discourse_adj' in config.feature_types:
            self.discourse_embed = nn.Embedding(len(DISCOURSE_RELATIONS)+1,1,padding_idx=0)
        if 'distance_adj' in config.feature_types:
            self.distance_embed = nn.Embedding(config.max_utt_num*2-1,1)
        if "cooccurrence_adj" in config.feature_types:
            self.cooccurrence_embed = nn.Embedding(5+1,1,padding_idx=0) # [0,1,2,3]
        
    def forward(self,query,key,value,adj_mats,mask):
        batch_size = query.shape[0]
        Q = self.fc_q(query) #bs,seq_len,hid_dim
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3) #[bs,n_heads,seq_len,hid_dim]
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q,K.permute(0,1,3,2)) / self.scale
        #energy = [batch size, n heads, query len, key len]
        
        adj_ls = []
        for k,v in adj_mats.items():
            if k=='distance_adj':
                adj_ls.append(self.distance_embed(v).squeeze(-1)) # [bs,num_utt,num_utt]
            elif k=='speaker_adj':
                adj_ls.append(v)  # [bs,num_utt,num_utt]
            elif k=='discourse_adj':
                adj_ls.append(self.discourse_embed(v).squeeze(-1))
            elif k=="cooccurrence_adj":
                adj_ls.append(self.cooccurrence_embed(v).squeeze(-1))
        # [bs,feature_map_num,num_utt,num_utt]
        feature_map = torch.stack(adj_ls,dim=1)
        
        
        # feature_conv_output: [bs,n_heads,q_len,k_len]
        feature_conv_output = self.conv_activation_fn(torch.stack([conv(feature_map).squeeze(1) for conv in self.feature_conv],dim=1))
        

        
        energy_ls = []
        for idx in range(self.n_heads):
            energy_ls.append(self.combine_conv(torch.stack([energy[:,idx,:,:],feature_conv_output[:,idx,:,:]],dim=1)).squeeze(1))
        energy = self.conv_activation_fn(torch.stack(energy_ls,dim=1)) # bs,n_head,q_len,k_len 
        
        if mask is not None:
            
            #_energy = _energy.masked_fill(mask==0,float("-inf"))
            energy = energy.masked_fill(mask==0,float("-inf"))
            #feature_conv_output = feature_conv_output.masked_fill(mask==0,float("-inf"))
        
        attention = torch.softmax(energy,dim=-1)
        #feature_conv_output = torch.softmax(feature_conv_output,dim=-1)
        #feature_conv_output = torch.softmax(feature_conv_output,dim=-1)
        #energy = [batch size, n heads, query len, key len]
        
        #p_dyna = torch.stack([torch.sigmoid(self.p_dyna_linear[x](Q[:,x,:,:])) for x in range(self.n_heads)],dim=1) # bs,n_heads,seq_len,1
        #attention = p_dyna * energy + (1-p_dyna) * feature_conv_output
        # if self.config.ablation_type == "static":
        #    attention = torch.softmax(feature_conv_output,dim=-1)
        # elif self.config.ablation_type == 'dynamic':
        #    attention = torch.softmax(_energy,dim=-1)

        x = torch.matmul(self.dropout(attention),V)
        x = x.permute(0,2,1,3).contiguous()
        x = x.view(batch_size,-1,self.hid_dim)
        x = self.fc_o(x)
        return x,attention

class GraphTransformerLayer(BartEncoderLayer):
    
    def __init__(self,config):
        super().__init__(config)
        self.self_attn = GraphTransformerMultiHeadAttentionLayer(config)
        self.fc1 = nn.Linear(self.embed_dim, config.gt_encoder_ffn_dim)
        self.fc2 = nn.Linear(config.gt_encoder_ffn_dim, self.embed_dim)
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
        **kwargs,
        ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        adj_mats = kwargs.get('adj_mats',None)
        residual = hidden_states
        hidden_states, attn_weights = self.self_attn(
            hidden_states,hidden_states,hidden_states,
            adj_mats,
            mask=attention_mask,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class GraphTransformer(BartEncoder):

    def __init__(self,config):
        super().__init__(config)
        self.layers = nn.ModuleList([GraphTransformerLayer(config) for _ in range(int(config.gt_encoder_layers))])
        del self.embed_positions
        if config.gt_pos_embed == 'learned':
            self.embed_positions = nn.Embedding(1026,config.d_model)
        elif config.gt_pos_embed == 'sinusoidal':
            self.embed_positions = SinusoidalPositionalEmbedding(1024,config.d_model)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        bs,num_utt = input_shape
        if hasattr(self,'embed_positions'):
            if isinstance(self.embed_positions,SinusoidalPositionalEmbedding):
                embed_pos = self.embed_positions(input_shape)
            else:
                embed_pos = self.embed_positions(torch.arange(num_utt).view(1,-1).repeat(bs,1).to(inputs_embeds.device))
            hidden_states = inputs_embeds + embed_pos
        else:
            hidden_states = inputs_embeds
    
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask =  (attention_mask == 1).unsqueeze(1).unsqueeze(2)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if getattr(self.config, "gradient_checkpointing", False) and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                        #dis_embed = self.dis_embed,
                        #speaker_embed = self.speaker_embed,
                        **kwargs,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

class RGCN(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.graph_layers = nn.ModuleList(
            [dglnn.RelGraphConv(
                in_feat = config.d_model,
                out_feat = config.d_model,
                num_rels = 17,
                regularizer = 'basis',
                self_loop = False,
            )
            for _ in range(3)]
            )
    
    def forward(self,inputs_embeds,attention_mask,**kwargs):
        input = inputs_embeds
        # input: bs,num_utt,num_utt
        # attn_mask: bs,num_utt
        adj_mats = kwargs.get('adj_mats',None) 
        discourse_adj = adj_mats['discourse_adj'] # bs,num_utt,num_utt
        graph_output = []
        for batch_idx in range(input.size()[0]):
            src,trg,etype = [],[],[]
            mat = discourse_adj[batch_idx].tolist()
            mat_len = len(mat)
            for i in range(mat_len):
                for j in range(mat_len):
                    if mat[i][j] != 0:
                        src.append(i)
                        trg.append(j)
                        etype.append(mat[i][j])
            graph = dgl.graph((torch.tensor(src),torch.tensor(trg)),num_nodes = mat_len).to(input.device)
            etype = torch.tensor(etype,device=input.device,dtype=torch.int64)
            f_in = input[batch_idx]
            for layer in self.graph_layers:
                f_in  = layer(graph,f_in,etype)
            graph_output.append(f_in)
        graph_output = torch.stack(graph_output,dim=0)
        assert graph_output.size() == input.size()
        return BaseModelOutput(last_hidden_state = graph_output)

