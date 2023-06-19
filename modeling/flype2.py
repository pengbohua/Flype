from abc import ABC
from transformers import Blip2Model, Blip2Config, OPTForSequenceClassification, OPTConfig
import torch
import json
from typing import Optional, Union, Tuple
from torch.nn.functional import normalize
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn as nn
from utils import convert_dtype_to_string
from modeling.soft_embedding import SoftPrompt


class FLYPE(Blip2Model, ABC):
    config_class = Blip2Config
    main_input_name = "pixel_values"
    learnable_param = {'query_tokens', 'soft_prompts.prompts', 'classification_head'}
    # learnable_param = {'query_tokens', 'soft_prompts.prompts', 'classification_head', 'norm',}

    def __init__(self, config: Blip2Config):
        super().__init__(config)

        lang_config = OPTConfig.from_pretrained('saleforce_opt')
        self.language_model = OPTForSequenceClassification(lang_config)
        self.config.pad_token_id = lang_config.pad_token_id
        self.config.num_attention_heads = config.text_config.num_attention_heads
        self.config.num_hidden_layers = lang_config.num_hidden_layers
        self.config.pr_seq_len = config.pr_seq_len

        self.dropout = torch.nn.Dropout(0.1)
        self.weight = torch.Tensor([config.weight]).cuda()
        for param in self.parameters():
            param.requires_grad = False

        for _name, param in self.named_parameters():
            for _tunable_param in self.learnable_param:
                if _tunable_param in _name:
                    param.requires_grad = True
                    break

        print("soft prompt length:", config.pr_seq_len)
        self.soft_prompts = SoftPrompt(self.language_model.model.decoder.embed_tokens, config.pr_seq_len, mode=config.emb_init_mode)
        self.language_model.set_input_embeddings(self.soft_prompts)          # set model embs as prompts

        mm_hidden_size = config.qformer_config.hidden_size + config.text_config.hidden_size
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=mm_hidden_size,
        #                                                 nhead=config.num_heads, dropout=0.1)
        # self.fusion_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.classification_head = nn.Linear(mm_hidden_size, 1)

        all_param = 0
        for _name, param in self.named_parameters():
            if param.requires_grad:
                print("name", _name)
                all_param += param.numel()

        print("Number of training parameters: {}M".format(all_param / 1000000))

    def _encode(self, query_tokens, attention_mask=None, past_key_values=None, image_embeds=None, image_atts=None, return_dict=True):
        """

        @param query_tokens: query token can be hard prompts / soft prompts / text
        @param attention_mask:  attention_mask of query tokens / query tokens + prefix (layerwise). set attention_mask = 0 means removing the multimodal linker
        @param past_key_values:   prefix (layerwise)
        @param image_embeds:
        @param image_atts:
        @param return_dict:
        @return:
        """
        assert image_embeds is not None
        batch_size = len(image_embeds)
        query_attention_mask = torch.ones((batch_size, query_tokens.shape[1]), device=query_tokens.device)
        query_outputs = self.qformer(
                query_embeds=query_tokens,
                attention_mask=query_attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=return_dict,
            )

        return query_outputs

    def forward(
        self,
            input_ids: torch.LongTensor,                        # texts
            pixel_values: torch.FloatTensor,                    # images
            labels: Optional[torch.LongTensor] = None,          # labels
            attention_mask: Optional[torch.LongTensor] = None,  # texts mask
            use_fusion_head: Optional[bool] = False,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = True,
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
        batch_size = len(pixel_values)
        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        mm_outputs = self._encode(
            query_tokens=query_tokens,
            image_embeds=image_embeds,
            image_atts=image_atts,
            return_dict=return_dict,
        )
        mm_embs = mm_outputs[0] if not return_dict else mm_outputs.last_hidden_state
        mm_embs = mm_embs[:, 0, :]

        # apply prompt tuning
        input_ids = torch.cat([torch.full((batch_size, self.config.pr_seq_len), 50256, device=input_ids.device), input_ids], dim=1)
        attention_mask = torch.cat([torch.full((batch_size, self.config.pr_seq_len), 1, device=input_ids.device), attention_mask], dim=1)
        text_outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        text_last_hidden_states = text_outputs.hidden_states[-1]
        sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(input_ids.device)
        pooled_text_embs = text_last_hidden_states[torch.arange(batch_size, device=input_ids.device), sequence_lengths, :]

        mm_feat = torch.cat([mm_embs, pooled_text_embs], dim=1)
        # mm_feat = self.fusion_encoder(mm_feat)

        mm_feat = self.dropout(mm_feat)
        logits = self.classification_head(mm_feat)
        if labels is None:
            return SequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=mm_outputs.hidden_states,
            attentions=mm_outputs.attentions,
        )
        else:
            loss_fct = nn.BCEWithLogitsLoss(weight=self.weight)
            loss = loss_fct(logits.squeeze(), labels.float())
            return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=mm_outputs.hidden_states,
            attentions=mm_outputs.attentions,
        )
