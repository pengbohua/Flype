import os
import argparse
from os.path import dirname
import logging
from torch.utils.data import DataLoader, Dataset
# from modeling.flype1 import FLYPE
# from transformers import AutoProcessor, BlipConfig
from modeling.flype2 import FLYPE
from transformers import AutoProcessor, Blip2Config
import torch
from format_checker.subtask_1 import check_format
from sklearn.metrics import accuracy_score, f1_score
from meta_dataset import Blip2MMDataset, collate
from utils import move_to_cuda
from transformers import Trainer, EarlyStoppingCallback, TrainingArguments, IntervalStrategy
import numpy as np
import wandb
from scorer.subtask_1 import evaluate


ROOT_DIR = dirname(dirname(__file__))
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_metrics(p):
    pred, labels = p
    pred = np.greater(pred, 0.5).squeeze()
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    # wandb.log({"val accuracy": accuracy})
    return {"accuracy": accuracy,}


def train_flype(data_dir, split, train_fpath, valid_fpath, args):
    """

    @param data_dir:
    @param split:
    @param train_fpath:
    @param test_fpath:
    @param results_fpath: results/
    @param model_id:
    """

    training_args = TrainingArguments(
        evaluation_strategy=IntervalStrategy.STEPS,  # "steps"
        eval_steps=50,
        output_dir='./checkpoints',
        num_train_epochs=5,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=64,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        metric_for_best_model='accuracy',
        load_best_model_at_end=True,
        fp16=True,      # fp16 automatic
    )

    # load model
    # processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")        # baseline
    # model_config = Blip2Config.from_pretrained("Salesforce/blip-itm-base-coco")
    model_config = Blip2Config.from_pretrained(args.backbone)
    model_config.pr_seq_len = args.pr_seq_len
    model_config.use_itm_head = args.use_itm_head
    model_config.weight = args.weight
    model_config.num_heads = args.heads
    model_config.emb_init_mode = args.emb_init_mode
    model_config.prefix_projection = True
    model = FLYPE(model_config).cuda()
    # model = torch.nn.DataParallel(model)

    # Dataset and DataLoader
    train_dataset = Blip2MMDataset(train_fpath, data_dir, pretrained_model_path=args.backbone, max_text_length=args.max_text_length)
    val_dataset = Blip2MMDataset(valid_fpath, data_dir, pretrained_model_path=args.backbone, max_text_length=args.max_text_length)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.02, )],
    )
    trainer.train()
    print("Saving checkpoint in checkpoints/")
    os.makedirs("checkpoints/flype_opt", exist_ok=True)
    model_state_dict = model.state_dict()
    light_state_dict = {}
    for k, v in model_state_dict.items():
        for _param in model.learnable_param:
            if _param in k:
                light_state_dict[k] = v
                break

    torch.save(light_state_dict, "checkpoints/flype_opt/bs_{}_lr{}_seq_len{}_epochs{}.pt".format(args.train_batch_size, args.lr, args.pr_seq_len, args.epochs))


def test_model(data_dir, split, test_fpath, results_fpath, model_id='flype', args=None):

    # model_config = Blip2Config.from_pretrained("Salesforce/blip-itm-base-coco")
    model_config = Blip2Config.from_pretrained(args.backbone)
    model_config.pr_seq_len = args.pr_seq_len
    model_config.weight = args.weight
    model_config.num_heads = args.heads
    model_config.emb_init_mode = args.emb_init_mode
    model = FLYPE(model_config).to(device)

    state_dict = torch.load("checkpoints/flype_opt/bs_{}_lr{}_seq_len{}_epochs{}.pt".format(args.train_batch_size, args.lr, args.pr_seq_len, args.epochs))
    model.load_state_dict(state_dict, strict=False)
    model = model.cuda()

    test_dataset = Blip2MMDataset(test_fpath, data_dir, pretrained_model_path=args.backbone, max_text_length=args.max_text_length) # dev test
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate)
    with open(results_fpath, "w") as results_file:
        results_file.write("tweet_id\tclass_label\trun_id\n")

        for i, batch_dict in enumerate(test_loader):
            tweet_id = test_dataset.tweet_ids[i]
            batch_dict.pop("labels")
            batch_dict = move_to_cuda(batch_dict)
            with torch.no_grad():
                outputs = model(**batch_dict)
                prd = torch.sigmoid(outputs['logits']).item()
            if prd > 0.35:
                label = "Yes"
            else:
                label = "No"

            results_file.write("{}\t{}\t{}\n".format(tweet_id, label, "{}".format(model_id)))

    gold_fpath = os.path.join(data_dir, f'{os.path.basename(test_fpath)}')
    # evaluation on dev
    if check_format(results_fpath):
        acc, precision, recall, f1 = evaluate(gold_fpath, results_fpath, subtask="A")
        logging.info(f"Qformer for Accuracy (positive class): {acc}")
        logging.info(f"Qformer for Precision (positive class): {precision}")
        logging.info(f"Qformer for Recall (positive class): {recall}")
        logging.info(f"Qformer for F1 (positive class): {f1}")
    with open("results/hypersearch_flype.txt", "a") as f:
        f.write("bs_{}_lr{}_seq_len{}_epochs{}\t {}".format(args.train_batch_size, args.lr, args.pr_seq_len, args.epochs, f1))


def main(config=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", "-d", required=False, type=str,
                        default="data/en/train_data",
                        help="The absolute path to the training data")
    parser.add_argument("--heads", required=False, type=float, default=8,
                        help="learning rate")
    parser.add_argument("--lr", required=False, type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument("--epochs", required=False, type=int, default=2,
                        help="epochs")
    parser.add_argument("--weight", required=False, type=float, default=2.0,
                        help="weight for imbalance classification")
    parser.add_argument("--file-name-train", required=False, type=str,
                        default="CT23_1A_checkworthy_multimodal_english_train.jsonl",
                        help="Input file name, exptects jsonl")

    parser.add_argument("--file-name-val", required=False, type=str,
                        default="CT23_1A_checkworthy_multimodal_english_dev.jsonl",
                        help="Input file name, exptects jsonl")

    parser.add_argument("--file-name-test", required=False, type=str,
                        default="CT23_1A_checkworthy_multimodal_english_test_gold.jsonl",
                        help="Input file name, exptects jsonl")

    parser.add_argument("--backbone", required=False, type=str,
                        default="Salesforce/blip2-opt-2.7b", help="backbone for the blip2. baseline blip-itm-base-coco")
    parser.add_argument("--out-file-name", "-o", required=False, type=str,
                        default="train_feats.json", help="Output feature file name")
    parser.add_argument("--lang", "-l", required=False, type=str, default="en",
                        help="Options: ar | en")
    parser.add_argument("--max-text-length", required=False, type=int, default=20,
                        help="max sequence length for text inputs")
    parser.add_argument("--use-itm-head", required=False, type=bool, default=True,
                        help="Use pretrained qformer or not")
    parser.add_argument("--pr-seq-len", required=False, type=int, default=5,
                        help="prefix sequence length")
    parser.add_argument("--emb-init-mode", required=False, type=str, default='sequential',
                        help="embedding initialization mode random: randomly initialize, "
                             "fixed: 50261, sampled: sampled from vocab embs")
    parser.add_argument("--return-dict", required=False, type=bool, default=False,
                        help="return dict during training. Turn it off to save VRAM")
    parser.add_argument("--pr-projection", required=False, type=bool, default=True,
                        help="project the prompts to a shared space")
    parser.add_argument("--train-batch-size", required=False, type=int, default=8,
                        help="training batch size")
    parser.add_argument("--test-batch-size", required=False, type=int, default=16,
                        help="test batch size")
    args = parser.parse_args()
    wandb.init(entity='marvinpeng', project="blipitm")
    # args.lr = config.lr
    # args.train_batch_size = config.train_batch_size
    # args.pr_seq_len = config.pr_seq_len
    # args.epochs = config.epochs
    train_flype(args.data_dir, 'valid', args.file_name_train, args.file_name_val, args)
    test_model("data/en/test_data", 'test', args.file_name_test, args.file_name_test,
               "results/flype_subtask1A_bs_{}_lr{}_seq_len{}_epochs{}.tsv".format(args.train_batch_size,
                                                args.lr, args.pr_seq_len, args.epochs,), args)

if __name__ == '__main__':
    main(wandb.config)