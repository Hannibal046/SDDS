from dataclasses import dataclass, field
import sys
import os
from typing import Optional
from torch.utils import data
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
import json
import os
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from transformers.debug_utils import DebugOption
from transformers.file_utils import (
    cached_property,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_available,
    is_torch_tpu_available,
    torch_required,
)
from transformers.trainer_utils import EvaluationStrategy, IntervalStrategy, SchedulerType, ShardedDDPOption
from transformers.utils import logging

logger = logging.get_logger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """  
    model_type:str='graphtransformer'
    
    # all_bart_base config
    ablation_type:str='no' # [no,static,dynamic,no_extra_encoder]
    utt_pooling:str = 'average'
    gt_activation_dropout:float = 0.1
    gt_activation_function: str = 'gelu'
    gt_add_bias_logits:bool = False
    gt_add_final_layer_norm:bool = False
    gt_attention_dropout:float = 0.1 
    gt_d_model:int = 768 
    gt_decoder_attention_heads:int = 12 
    gt_decoder_ffn_dim:int = 3072 
    gt_decoder_layerdrop:float = 0.0
    gt_decoder_layers:int = 6 
    gt_dropout:float = 0.1 
    gt_encoder_attention_heads:int = 12 
    gt_encoder_ffn_dim:int = 3072 
    gt_encoder_layerdrop :float= 0.0
    gt_encoder_layers :int= 6 
    gt_init_std :float= 0.02 
    gt_is_encoder_decoder:bool = True 
    gt_normalize_before :bool= False 
    gt_normalize_embedding :bool= True 
    gt_scale_embedding :bool= False
    num_beams :int= 5
    max_target_length :int= 100
    min_target_length :int= 5
    max_utt_num :int= 47
    decoder_start_token_id:int = 0
    gt_pos_embed:str = ''
    backbone_model: str = field(
        default=None,
        metadata={"help": "bart_base/bart_large"}
    )
    feature_types: List[str] = field(
        default_factory=lambda: ['speaker_adj','distance_adj','discourse_adj'],
        metadata={"help":"addtional graph info to be fused into the graphtrans"}
    )
    rezero: int = field(
        default = -1,
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
        
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    exp_description: str = field(
        default=None,
        metadata={"help":"用来对实验进行描述"}
    )

    max_seq_len_per_utt: int = field(
        default=100,
        metadata = {"help":"max_seq_len_per_utt"}
    )
    reprocess_data: bool = field(
        default=False, metadata={"help": "whether to reprocess data into arrow format"}
    )
    dataset_cached_path: Optional[str] = field(
        default=None, metadata={"help": "The path of the dataset to cache at."}
    )
    save_dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The path of the dataset to load from."}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    speaker_column: Optional[str] = field(
        default=None,
        metadata={"help":"The name of the column in the datasets containing the speakers (for summarization)."}
    )
    unique_speaker_column: Optional[str] = field(
        default=None,
        metadata={"help":"The name of the column in the datasets containing the unique speakers (for summarization)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
        
    val_min_target_length: Optional[int] = field(
        default=3,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )

    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )


def check_args(data_args,model_args,training_args):

    '''
    this a function to detect whether the args are valid and has no controversy among themselves
    '''
    data_args.num_beams = model_args.num_beams
    data_args.val_max_target_length = model_args.max_target_length
    data_args.val_min_target_length = model_args.min_target_length
    data_args.max_target_length = model_args.max_target_length
    data_args.max_utt_num = model_args.max_utt_num
    if 'samsum' in data_args.train_file:
        data_args.speaker_column = data_args.unique_speaker_column
    if training_args.no_cuda:
        training_args.fp16 = False
    return data_args,model_args,training_args