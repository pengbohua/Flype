from transformers import Blip2Processor, Blip2Model
from torch.utils.data import Dataset
import jsonlines
import os
import torch
from preprocessing.TweetNormalizer import normalizeTweet
from PIL import Image


def collate(batch_dict):
    input_ids = []
    attention_masks = []
    labels = []
    pixel_values = []
    ids = []
    for b_dict in batch_dict:
        input_ids.append(b_dict['input_ids'])
        attention_masks.append(b_dict['attention_mask'])
        labels.append(b_dict['label'])
        pixel_values.append(b_dict['pixel_values'])
        ids.append(b_dict['id'])
    return {'id': ids,
            'pixel_values': torch.cat(pixel_values, 0),
            'input_ids': torch.cat(input_ids, 0),
            'attention_mask': torch.cat(attention_masks, 0),
            'labels': torch.stack(labels, 0)
            }


class Blip2MMDataset(Dataset):
    def __init__(self, jsonl_file, data_dir, pretrained_model_path, max_text_length):
        self.jsonl_file = [obj for obj in jsonlines.open(os.path.join(data_dir, jsonl_file))]
        self.data_dir = data_dir
        self.max_text_length = max_text_length
        self.processor = Blip2Processor.from_pretrained(pretrained_model_path)

    def __len__(self):
        return len(self.jsonl_file)

    def _convert_label(self, _label):
        if _label == 'Yes':
            converted_label = 1
        elif _label == 'No':
            converted_label = 0
        else:
            raise ValueError
        return torch.tensor(converted_label).long()

    def __getitem__(self, idx):
        obj = self.jsonl_file[idx]
        tweet_id = obj["tweet_id"]
        image_path = os.path.join(self.data_dir, obj["image_path"])
        img = Image.open(image_path).convert("RGB")
        text = normalizeTweet(obj["tweet_text"])
        inputs = self.processor(images=[img],
                       text=[text],
                       return_tensors="pt",
                       padding='max_length',
                       max_length=self.max_text_length,
                       truncation=True
                       )
        label = obj["class_label"]
        label = self._convert_label(label)
        return {"id": tweet_id, "pixel_values": inputs['pixel_values'], "input_ids": inputs['input_ids'], "attention_mask": inputs['attention_mask'], "label": label}


if __name__ == '__main__':
    data_dir = "data/en/train_data"
    train_file = "CT23_1A_checkworthy_multimodal_english_train.jsonl"
    blip2dataset = Blip2MMDataset(train_file, data_dir, pretrained_model_path="Salesforce/blip2-opt-2.7b")
    loader = iter(blip2dataset)
    print(next(loader))