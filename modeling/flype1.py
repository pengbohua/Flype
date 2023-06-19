from abc import ABC
from transformers import BlipForImageTextRetrieval, BlipConfig
import torch
from typing import Optional, Union, Tuple
from torch.nn.functional import normalize
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn


class PrefixEncoder(torch.nn.Module):
    '''
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.prefix_hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.prefix_hidden_size, config.prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.prefix_hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_hidden_layers * 2 * config.prefix_hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


class FLYPE(BlipForImageTextRetrieval, ABC):
    config_class = BlipConfig
    main_input_name = "pixel_values"
    learnable_param = {'prefix', 'norm'}
    backbone = "Salesforce/blip-itm-base-coco"

    def __init__(self, config: BlipConfig):
        super().__init__(config)
        self.pre_seq_len = config.prefix_seq_len
        print("prefix length", self.pre_seq_len)
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.hidden_size = config.projection_dim

        self.config.num_attention_heads = config.text_config.num_attention_heads
        self.config.num_hidden_layers = config.text_config.num_hidden_layers
        self.config.pre_seq_len = config.prefix_seq_len
        self.config.prefix_hidden_size = config.text_config.hidden_size
        self.config.prefix_projection = config.prefix_projection
        self.prefix_encoder = PrefixEncoder(self.config)
        self.dropout = torch.nn.Dropout(0.1)
        print(self.prefix_encoder)

        for param in self.parameters():
            param.requires_grad = False

        for _name, param in self.named_parameters():
            for _tunable_param in self.learnable_param:
                if _tunable_param in _name:
                    param.requires_grad = True
                    break

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=16, dropout=0.1)
        self.fusion_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        if config.use_itm_head:
            self.classification_head = nn.Linear(self.config.prefix_hidden_size, 1)
        else:
            self.classification_head = nn.Linear(self.hidden_size, 1)

        all_param = 0
        for _, param in self.named_parameters():
            if param.requires_grad:
                all_param += param.numel()

        print("Number of training parameters: {}M".format(all_param / 1000000))

    def get_prompt(self, batch_size, prefix_tokens):
        """Get prompt tokens to meet input requirements.

        Args:
            batch_size (_type_): _description_

        Returns:
            _type_: _description_
        """
        prefix_tokens = prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.text_encoder.device)
        past_key_values = self.prefix_encoder(prefix_tokens)

        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.config.num_hidden_layers * 2,
            self.config.num_attention_heads,
            self.config.prefix_hidden_size // self.config.num_attention_heads
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)     # layers, bs, num_heads, seq_len, hid_dim
        return past_key_values, prefix_tokens

    def _encode(self, input_ids, attention_mask, past_key_values, image_embeds=None, image_atts=None, return_dict=True):
        prefix_mask = torch.ones(len(attention_mask), self.pre_seq_len).long().to(attention_mask.device)
        prefix_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=prefix_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            past_key_values=past_key_values,
            return_dict=return_dict,
        )
        return outputs

    def forward(
        self,
            input_ids: torch.LongTensor,                        # text
            pixel_values: torch.FloatTensor,                    # image
            use_itm_head: Optional[bool] = False,
            attention_mask: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        Returns:

        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions   # output attention for visualization
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states          # output hidden states
        )
        # step 0: get the prompts
        batchsize = len(pixel_values)
        past_key_values, prefix_tokens = self.get_prompt(batchsize, self.prefix_tokens)

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long)

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        if use_itm_head:
            outputs = self._encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_embeds=image_embeds,
            past_key_values=past_key_values,
            image_atts=image_atts,
            return_dict=return_dict,
        )
            output = outputs[0] if not return_dict else outputs.last_hidden_state
            sim = self.itm_head(output[:, 0, :])    # a head for similarity estimation
            output = output[:, 0, :]
        else:
            text_outputs = self._encode(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                return_dict=return_dict,
            )
            text_outputs = text_outputs if not return_dict else text_outputs.last_hidden_state
            image_feat = self.vision_proj(image_embeds[:, 0, :])
            text_feat = self.text_proj(text_outputs[:, 0, :])
            mm_feat = torch.cat([image_feat, text_feat], dim=1)
            output = self.fusion_encoder(mm_feat)

        output = self.dropout(output)
        logits = self.classification_head(output)

        return SequenceClassifierOutput(
            logits=logits,
            hidden_states=output_hidden_states,
            attentions=output_attentions,
        )
