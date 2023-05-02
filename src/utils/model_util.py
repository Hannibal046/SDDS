from torch.utils import data
from transformers import AutoConfig,AutoModel,AutoTokenizer
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from transformers.models.bart import BartConfig
from model import bart_model,graphtransformer
from dataclasses import asdict

def get_model_tokenizer(model_args,data_args):

    tokenizer = AutoTokenizer.from_pretrained(model_args.backbone_model,use_fast=model_args.use_fast_tokenizer)
    config = graphtransformer.GraphTransformerConfig(**asdict(model_args))
    
    model = bart_model.BART.from_pretrained(model_args.backbone_model,config=config)
    model.config.output_attentions = True
    if '<sep>' not in tokenizer.additional_special_tokens:
        special_tokens_dict = {"additional_special_tokens":["<sep>"]}#,"#Person2#","#Person1#","#Person3#","#Person4#","#Person5#","#Person6#"]}
        tokenizer.add_special_tokens(special_tokens_dict)
    return model,tokenizer
