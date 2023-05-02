import json
import math
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# my own module
from utils.training_util import ShowModelParamsCallBack,FineTuneCallBack #have to import comet_ml before torch


from torch.utils import data

import sys
from dataclasses import dataclass, field,asdict
from typing import Optional

import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from transformers.optimization import AdamW
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

# my own module
from utils.args_util import ModelArguments,DataTrainingArguments,check_args
from _transformers.data_collator import MyDataCollatorForSeq2Seq
from _transformers.seq2seq_trainer import Seq2SeqTrainer
from _transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from utils import model_util,data_util,training_util
from utils.CONSTANT import *
import logging
from utils.metrics_util import get_bert_score,get_rouge_score,get_meteor_score

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.7.0.dev0")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)




def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    config_file = './config/graphbart_config.json'
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):config_file = sys.argv[1]
    model_args, data_args, training_args = parser.parse_json_file(config_file)
    data_args,model_args,training_args = check_args(data_args,model_args,training_args)

    #save config file
    output_dir = training_args.output_dir
    if not os.path.isdir(output_dir):
        os.system(f"mkdir {output_dir}")
    os.system(f"cp {config_file} {output_dir}/run_config.json")
    
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    # ## add log to file
    os.makedirs(training_args.output_dir,exist_ok=True)
    os.system(f'cp -r model {training_args.output_dir}')
    os.system(f'cp -r utils {training_args.output_dir}')

    logging.basicConfig(
        format="%(asctime)s-%(levelname)s-%(name)s%(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    #logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    # log args
    args_dict = {"data_args":data_args,"model_args":model_args,"training_args":training_args}
    keys_ls = list(asdict(training_args).keys()) + list(asdict(model_args).keys()) + list(asdict(data_args).keys())
    max_width = max([len(arg_name) for arg_name in keys_ls])
    for k,v in args_dict.items():
        logger.info("*"*SCREEN_WIDTH)
        logger.info(k)
        for arg_name,arg_value in asdict(v).items():
            logger.info(f"{arg_name:{max_width}}  {arg_value}")


    # Set seed before initializing model.
    set_seed(training_args.seed)

    model,tokenizer = model_util.get_model_tokenizer(model_args,data_args)
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    # get_train_val_test_dataset
    raw_dataset = data_util.get_dataset(data_args,model_args)
    train_dataset,eval_dataset,predict_dataset = data_util.data_preprocessing(raw_dataset,tokenizer,training_args,data_args,model_args)
    model.resize_token_embeddings(len(tokenizer))
    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    
    data_collator = MyDataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        dis_offset = int(model.config.max_utt_num-1),
        adj_mat_ls = model_args.feature_types
    )


    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = {}
        metrics_ls = [get_rouge_score]#,get_bert_score,get_meteor_score]
        for metrics in metrics_ls:
            res = metrics(decoded_preds,decoded_labels)
            result.update(res)
        # keys: rouge-1(f,p,r),rouge-2,rouge-l,bert_p,bert_r,bert_f,meteor
        # Extract a few results from ROUGE
        result['rouge-1'] = result['rouge-1']['f'] * 100
        result['rouge-2'] = result['rouge-2']['f'] * 100
        result['rouge-l'] = result['rouge-l']['f'] * 100

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    callbacks_ls = []
    if training_args.comet_enable:
        callbacks_ls.append(CometCallBack(training_args))
    if training_args.freeze_fine_tune_enable:
        callbacks_ls.append(FineTuneCallBack(model_args))
    #callbacks_ls.append(ShowModelParamsCallBack)

    # bart_params,additional_params = [],[]
    # for name,p in model.named_parameters():
    #     if name.startswith('model.encoder.graphtransformer') or 'second_crossattn' in name:
    #         additional_params.append(p)
    #     else:
    #         bart_params.append(p)
    # optimizer = AdamW(
    #     [{'params':bart_params},
    #      {'params':additional_params,'lr':2*training_args.learning_rate}],lr= training_args.learning_rate
    # )
    
    # num_update_steps_per_epoch = len(train_dataset) // training_args.gradient_accumulation_steps
    # num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    # max_steps = math.ceil(training_args.num_train_epochs * num_update_steps_per_epoch)
    # scheduler = transformers.get_linear_schedule_with_warmup(optimizer, training_args.warmup_steps, max_steps, last_epoch=- 1)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        callbacks = callbacks_ls,
        #optimizers = (optimizer,scheduler)
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(
            max_length=data_args.val_max_target_length, num_beams=data_args.num_beams, metric_key_prefix="eval"
        )
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))
    
    # 清除所有的checkpoint
    file_ls = os.listdir(training_args.output_dir)
    for file in file_ls:
        if file.startswith('checkpoint'):
            os.system(f'rm -rf {os.path.join(training_args.output_dir,file)}')

    all_results_dir = os.path.join(training_args.output_dir,"all_results.json")
    best_rouge = json.load(open(all_results_dir))["eval_rouge-1"]
    
    log_dir = "/".join(training_args.output_dir.split('/')[:-1])
    
    os.system(f"mv {training_args.output_dir} {os.path.join(log_dir,str(best_rouge))}")
    return results


if __name__ == "__main__":
    main()
