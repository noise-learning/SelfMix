from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
import logging, sys, os, random
import torch
from datasets import *
from model import *
from trainer import *

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
    model_name_or_path: Optional[str] = field(
        default='bert-base-uncased',
        metadata={
            "help": "The pretrained model checkpoint for weights initialization."
        },
    )
    pretrained_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Bert4Classify model path."
        },
    )
    dropout_rate: float = field(
        default=0.1,
        metadata={"help": "Dropout rate"}
    )
    

@dataclass
class DataEvalArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of dataset"}
    )
    eval_file_path: Optional[str] = field(
        default=None,
        metadata={"help": "The eval data file (.csv)"}
    )
    batch_size: int = field(
        default=32,
        metadata={"help": "Batch size"}
    )
    max_sentence_len: Optional[int] = field(
        default=256,
        metadata={
            "help": "The maximum total input sentence length after tokenization. Sequences longer."
        },
    )

    
def main():
    parser = HfArgumentParser((ModelArguments, DataEvalArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()
        
    logger.info("Model Parameters %s", model_args)
    logger.info("Data Parameters %s", data_args)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    eval_datasets, eval_num_classes = load_dataset(data_args.eval_file_path, data_args.dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_model_name_or_path)
    selfmix_eval_data = SelfMixData(data_args, eval_datasets, tokenizer)
    
    model = Bert4Classify(model_args.pretrained_model_name_or_path, model_args.dropout_rate, eval_num_classes)
    model.load_model(model_args.model_name_or_path)
    
    tester = SelfMixTrainer(
        model=model,
        eval_data=selfmix_eval_data
    )
    tester.evaluate()


if __name__ == "__main__":
    main()
