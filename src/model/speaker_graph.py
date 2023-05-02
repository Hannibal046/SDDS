import dgl
import dgl.nn as dglnn
from networkx.readwrite.json_graph import adjacency
from transformers.models.bart.modeling_bart import BartEncoderLayer
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional,Tuple


@dataclass
class GraphOutput:
    """
    This is GraphOutput for transformer decoder to attend to
    """
    speaker_hidden_states:torch.FloatTensor = None #last hidden states, [bs,speaker_num,hid_dim]
    speaker_attentions:Optional[Tuple[torch.FloatTensor]] = None
    speaker_attention_mask:torch.FloatTensor = None


class SpeakerGraph(torch.nn.Module):

    def __init__(self,config):
        super().__init__()
        self.speaker_graph_type = config.speaker_graph_type
        if config.speaker_graph_type == "bart_encoder":
            self.speaker_graph = BartEncoderLayer(config)
        elif config.speaker_graph_type == "gcn":
            self.speaker_graph = GCN(config)
        elif config.speaker_graph_type == 'gat':
            self.speaker_graph = GAT(config)
        elif config.speaker_graph_type == 'graphsage':
            self.speaker_graph = GraphSage(config)
        elif config.speaker_graph_type == 'rgcn':
            self.speaker_graph = RGCN(config) 
    def forward(self,
                hidden_states,
                output_attentions=False,  
                **kwargs):
        """
        输入:
        hidden_states: [bs,seq_len,hid_dim]
        speaker_attention_mask: [bs,seq_len]
        kwargs:
            -expaned_speaker_attention_mask
            -speaker_attention_mask
        输出:
            -speaker_hidden_state
            -speaker_hidden_mask
        """
        if self.speaker_graph_type == 'bart_encoder':
            expanded_speaker_attention_mask = kwargs['expanded_speaker_attention_mask']
            output = self.speaker_graph(
                hidden_states,
                expanded_speaker_attention_mask = expanded_speaker_attention_mask,
                output_attentions=output_attentions,   
            )

            return GraphOutput(
                speaker_hidden_states=output[0],
                speaker_attentions=output[2] if output_attentions else None
            )
        
        """
        if speaker_graph_type is not bart_encoder
        we have to:
        1. define the graph data structure
        2. get graph_transformed hidden states
        3. return new attentino mask for decoder
        """

        # do not use graph_batch for now
        
        # extract speaker hidden states
        edges = kwargs['edges'] #[[(),()],....] #这里需要考虑到截断问题,因为edges的计算是没有截断的,但是输入到bart模型里面的输入是有截断的        

        speaker_attention_mask  = kwargs['special_tokens_mask'] #[bs,seq_len]
        batch_size,seq_len = speaker_attention_mask.shape
        
        speaker_nums_per_instance = torch.sum(speaker_attention_mask,dim=1,keepdim=False)
        
        hidden_states = hidden_states[speaker_attention_mask.bool()] # hidden_states [allbatch_speaker_nums,hid_dim]
        
        # calculate graph output by gnn
        pt = 0
        graph_outputs = []
        for batch_idx in range(batch_size):
            graph_input = hidden_states[pt:pt+speaker_nums_per_instance[batch_idx]] #[speaker_nums_per_batach,hid_dim]
            pt += speaker_nums_per_instance[batch_idx]
            
            # 处理truncation的问题
            num_turns = graph_input.shape[0]
            for key,value in edges[batch_idx].items():
                u,v = value
                num_edges = len(u)
                assert len(value[0]) == len(value[1])
                for i in range(num_edges):
                    if u[i] >= num_turns:
                        u[i] = -1
                        v[i] = -1
                    if v[i] >= num_turns:
                        u[i] = -1
                        v[i] = -1
                edges[batch_idx][key][0] = [x for x in u if x != -1]
                edges[batch_idx][key][1] = [x for x in v if x != -1]
        
            if self.speaker_graph_type != 'rgcn':
                new_edges = []
                for edge in edges:
                    ls = []
                    for k,v in edge.items():
                        ls += list(zip(v[0],v[1]))
                new_edge = [[],[]]
                for p in set(ls):
                    new_edge[0].append(p[0])
                    new_edge[1].append(p[1])
                new_edges.append(new_edge)
                edges = new_edges
                graph = dgl.graph(
                    edges[batch_idx],num_nodes = graph_input.shape[0]
                )
                graph_output = self.speaker_graph(graph,graph_input) # graph_output:[speaker_number,graph_hid_dim]
            else:
                edge = edges[batch_idx]
                etype = []
                new_edge = [[],[]]
                pt_ = 0
                for k,v  in edge.items():
                    etype += [pt_]*len(v[0])
                    new_edge[0] += v[0] 
                    new_edge[1] += v[1]
                    pt_ += 1
                etype = torch.tensor(etype,device=graph_input.device,dtype=torch.int64)
                graph = dgl.graph((new_edge[0],new_edge[1]),num_nodes=graph_input.shape[0]).to(graph_input.device)
                graph_output = self.speaker_graph(graph,graph_input,etype)
            graph_outputs.append(graph_output)
        
        # graph_outputs: bs,speaker_numbers,graph_hid_dim
        max_speaker_num = max([x.shape[0] for x in graph_outputs])
        graph_hid_dim = graph_outputs[0].shape[1]

        # make new attention mask
        # 1:not mask 0:mask
        attention_mask = torch.zeros((batch_size,max_speaker_num)).to(hidden_states.device)
        for batch_idx in range(batch_size):
            attention_mask[batch_idx,:graph_outputs[batch_idx].shape[0]] = 1
        
        # make new outputs
        ret = torch.zeros((batch_size,max_speaker_num,graph_hid_dim)).to(hidden_states.device)
        for batch_idx in range(batch_size):
            ret[batch_idx,:graph_outputs[batch_idx].shape[0],:] = graph_outputs[batch_idx]


        return GraphOutput(
                speaker_hidden_states = ret,
                speaker_attention_mask = attention_mask,
                )



class GCN(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.graph_layers = nn.ModuleList([dglnn.GraphConv(
                                            in_feats = config.d_model,
                                            output_feats = config.d_model,
                                            aggregate_type = config.aggregate_type) 

                                            for _ in range(config.num_graph_layers)])
        def forward(self,g,in_feat):
            h = in_feat
            for layer in self.graph_layers:
                h = layer(g,h,edge_weight=None)
            return h

class GraphSage(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.graph_layers = nn.ModuleList([dglnn.SAGEConv(
                                            in_feats = config.d_model,
                                            output_feats = config.d_model,
                                            aggregate_type = config.aggregate_type) 

                                            for _ in range(config.num_graph_layers)])
    def forward(self,g,in_feat):
        h = in_feat
        for layer in self.graph_layers:
            h = layer(g,h,edge_weight=None)
        return h

class GAT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.graph_layers = nn.ModuleList([dglnn.GATConv(
                                            in_feats = config.d_model,
                                            output_feats = config.d_model,
                                            num_heads = config.gat_num_heads
                                            ) 
                                          for _ in range(config.num_graph_layers)])
    
    def forward(self,g,in_feat):
        h = in_feat
        for layer in self.graph_layers:
            h = layer(g,h)
        return h

class RGCN(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        self.graph_layers = nn.ModuleList([dglnn.RelGraphConv(
                                            in_feat = config.d_model,
                                            out_feat = config.d_model,
                                            num_rels = len(config.relation_type),
                                            regularizer = 'basis',
                                            ) 
                                            for _ in range(config.num_graph_layers)])
    
    def forward(self,g,in_feat,etype):
        h = in_feat
        for layer in self.graph_layers:
            h = layer(g,h,etype)
        return h
