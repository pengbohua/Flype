import torch
import torch.nn as nn
import numpy as np


class SoftPrompt(nn.Module):
    def __init__(self,
                 wte: nn.Embedding,
                 prompt_length: int = 10,
                 mode: str = 'fixed'):
        """appends learned embedding to

        Args:
            wte (nn.Embedding): original transformer word embedding
            prompt_length (int, optional): number of tokens for task. Defaults to 10.
            mode (str, optional): initialization select from {'random': randomly initialized,
                                   'fixed': "madeupword0000": 50261,
                                   'sampled': sampled from vocab embs}
        """
        super(SoftPrompt, self).__init__()
        self.wte = wte
        self.prompt_length = prompt_length
        self.mode = mode
        self.prompts = self.initialize_embedding()

    def initialize_embedding(self,):
        """initializes learned embedding

        Args:
            same as __init__

        Returns:
            torch.float: initialized using original schemes
        """
        _vocab_size, _hidden_dim = self.wte.weight.shape
        if self.mode == 'fixed':
            return nn.Parameter(self.wte.weight[50261].expand(self.prompt_length, _hidden_dim).clone().detach())
        elif self.mode == 'random':
            return torch.nn.Embedding(self.prompt_length, _hidden_dim)
        elif self.mode == 'sampled':
            _idx = np.random.choice(list(range(_vocab_size)), self.prompt_length)
            return nn.Parameter(self.wte.weight[_idx, :].clone().detach())
        elif self.mode == 'sequential':
            return nn.Parameter(self.wte.weight[:self.prompt_length].clone().detach())
        else:
            raise NotImplementedError

    def forward(self, tokens):
        """run forward pass

        Args:
            tokens (torch.long): input tokens before encoding

        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        assert len(tokens.size()) == 2, "Make sure the tokenizer is loaded correctly."
        input_embedding = self.wte(tokens[:, self.prompt_length:])
        prompts = self.prompts.unsqueeze(0).repeat(input_embedding.size(0), 1, 1)
        return torch.cat([prompts, input_embedding], 1)


if __name__ == '__main__':
    from transformers import OPTForSequenceClassification, OPTConfig, AutoTokenizer

    config = OPTConfig.from_pretrained('meta_opt')
    model = OPTForSequenceClassification(config)
    # print(model)
    prompt_embs = SoftPrompt(model.model.decoder.embed_tokens, prompt_length=5)
    model.set_input_embeddings(prompt_embs)

    tokenizer = AutoTokenizer.from_pretrained('meta_opt')
    inputs = tokenizer("Visual language understanding is cool!", return_tensors="pt")
    n_tokens = 5
    # need to pad attention_mask and input_ids to be full seq_len + n_learned_tokens
    # even though it does not matter what you pad input_ids with, it's just to make HF happy
    inputs['input_ids'] = torch.cat([torch.full((1, n_tokens,), 50256), inputs['input_ids']], 1)
    inputs['attention_mask'] = torch.cat([torch.full((1, n_tokens,), 1), inputs['attention_mask']], 1)
    outputs = model(**inputs)
    print(outputs.logits.shape)
    print("Finish sanity checking")