# GPT-2 XL wrapper using HuggingFace tokenizer (BPE)
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import softmax


class GPT2XL():
    def __init__(self, model_name="gpt2-xl", device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).eval().to(self.device)

        vocab_dict = self.tokenizer.get_vocab()
        # self.vocab maps from id to token
        self.vocab = [k for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])]
        # self.word2id maps from token str
        self.word2id = self.tokenizer.get_vocab()
        self.UNK_ID = self.tokenizer.eos_token_id

    def encode(self, words):
        # Map words to IDs using first BPE token
        # this is a bit of a simplification as a word like "unhappiness" would be encoded with the token for "un"
        # probably doesn't have much of an effect for most words but is a limitation that GPT-1 doesn't have
        return [self.tokenizer.encode(" " + x, add_special_tokens=False)[0] for x in words]

    def get_story_array(self, words, context_words):
        # Get word ids for each phrase in a stimulus story
        nctx = context_words + 1
        story_ids = self.encode(words)
        story_array = np.zeros([len(story_ids), nctx]) + self.UNK_ID
        for i in range(len(story_array)):
            segment = story_ids[i:i+nctx]
            story_array[i, :len(segment)] = segment
        return torch.tensor(story_array).long()

    def get_context_array(self, contexts):
        # Get word ids for each context
        context_array = np.array([self.encode(words) for words in contexts])
        return torch.tensor(context_array).long()

    def get_hidden(self, ids, layer):
        # Get hidden layer representations
        mask = torch.ones(ids.shape).int()
        with torch.no_grad():
            outputs = self.model(input_ids=ids.to(self.device),
                                 attention_mask=mask.to(self.device),
                                 output_hidden_states=True)
        return outputs.hidden_states[layer].detach().cpu().numpy()

    def get_probs(self, ids):
        # Get next word probability distributions
        mask = torch.ones(ids.shape).int()
        with torch.no_grad():
            outputs = self.model(input_ids=ids.to(self.device),
                                 attention_mask=mask.to(self.device))
        probs = softmax(outputs.logits, dim=2).detach().cpu().numpy()
        return probs
