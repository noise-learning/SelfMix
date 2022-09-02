from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
import logging, sys, os, random
import torch
import numpy as np
from datasets import *
from model import *
from trainer import SelfMixTrainer

from transformers import (
    AutoModel,
    HfArgumentParser,
    set_seed,
    AutoTokenizer,
)


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune. 
    """
    
    # Huggingface's original arguments
    pretrained_model_name_or_path: Optional[str] = field(
        default='bert-base-uncased',
        metadata={
            "help": "The pretrained model checkpoint for weights initialization."
        },
    )
    dropout_rate: float = field(
        default=0.1,
        metadata={"help": "Dropout rate"}
    )
    
    # SelfMix's arguments
    p_threshold: float = field(
        default=0.5,
        metadata={"help": "Clean probability threshold"}
    )
    temp: float = field(
        default=0.5,
        metadata={"help": "Temperature for sharpen function"}
    )
    alpha: float = field(
        default=0.75,
        metadata={"help": "Alpha for beta distribution"}
    )
    lambda_p: float = field(
        default=0.2,
        metadata={"help": "Weight for Pseudo Loss"}
    )
    lambda_r: float = field(
        default=0.3,
        metadata={"help": "Weight for R-Drop loss"}
    )
    class_reg: bool = field(
        default=False,
        metadata={"help": "Whether to apply class regularization to loss"}
    )
    ## gmm arguments
    gmm_max_iter: int = field(
        default=10,
        metadata={"help": "The number of EM iterations to perform"}
    )
    gmm_tol: float = field(
        default=1e-2,
        metadata={"help": "The convergence threshold"}
    )
    gmm_reg_covar: float = field(
        default=5e-4,
        metadata={"help": "Non-negative regularization added to the diagonal of covariance."}
    )
    

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of dataset"}
    )
    train_file_path: Optional[str] = field(
        default=None,
        metadata={"help": "The train data file (.csv)"}
    )
    eval_file_path: Optional[str] = field(
        default=None,
        metadata={"help": "The eval data file (.csv)"}
    )
    batch_size: int = field(
        default=32,
        metadata={"help": "Batch size"}
    )
    batch_size_mix: int = field(
        default=16,
        metadata={"help": "Batch size for mix train"}
    )
    max_sentence_len: Optional[int] = field(
        default=256,
        metadata={
            "help": "The maximum total input sentence length after tokenization. Sequences longer."
        },
    )


@dataclass
class OurTrainingArguments:
    seed: Optional[int] = field(
        default=1,
        metadata={"help": "Seed"}
    )
    warmup_strategy: Optional[str] = field(
        default=None,
        metadata={
            "help": "Warmup strategy"
                  "no: no warmup before training"
                  "epoch: apply warmup-epoch stratrgy"
                  "samples: apply warmup-samples strategy"
        }
    )
    warmup_epochs: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of epochs to warmup the model"
            "only one of the warmup_epochs and warmup_samples should be specified"
        }
    )
    warmup_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of samples to warmup the model"
            "only one of the warmup_epochs and warmup_samples should be specified"
        }
    )
    train_epochs: int = field(
        default=4,
        metadata={"help": "Mix-up training epochs"}
    )
    lr: float = field(
        default=1e-5,
        metadata={"help": "Learning rate"}
    )
    grad_acc_steps: int = field(
        default=1,
        metadata={"help": "Gradient accumulation step"}
    )
    model_save_path: Optional[str] = field(
        default=None,
        metadata={"help": "The path to save model"}
    )

    
def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    logger.info("Model Parameters %s", model_args)
    logger.info("Data Parameters %s", data_args)
    logger.info("Training Parameters %s", training_args)
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    set_seed(training_args.seed)
    
    # load data
    train_datasets, train_num_classes = load_dataset(data_args.train_file_path, data_args.dataset_name)
    eval_datasets, eval_num_classes = load_dataset(data_args.eval_file_path, data_args.dataset_name)
    assert train_num_classes == eval_num_classes
    model_args.num_classes = train_num_classes
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_model_name_or_path)
    selfmix_train_data = SelfMixData(data_args, train_datasets, tokenizer)
    selfmix_eval_data = SelfMixData(data_args, eval_datasets, tokenizer)
    
    # load model
    model = Bert4Classify(model_args.pretrained_model_name_or_path, model_args.dropout_rate, model_args.num_classes)
    
    # build trainer
    trainer = SelfMixTrainer(
        model=model,
        train_data=selfmix_train_data,
        eval_data=selfmix_eval_data,
        model_args=model_args,
        training_args=training_args
    )
    
    # train and eval
    trainer.warmup()
    trainer.train()
    trainer.save_model()


if __name__ == '__main__':
    main()
    