import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import softmax

class GPT():    
    """wrapper for HuggingFace Causal LMs"""
    def __init__(self, model_name, device = 'cpu'): 
        self.device = device

        # Load model and tokenizer from HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).eval().to(self.device)

        # set vocab based on tokenzier
        vocab_dict = self.tokenizer.get_vocab()
        self.vocab = [k for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])]
        self.word2id = self.tokenizer.get_vocab()
        self.UNK_ID = self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else self.tokenizer.eos_token_id

    def encode(self, words):
        """map from words to ids using the tokenizer
        """
        return [self.tokenizer.encode(" " + x, add_special_tokens=False)[0] for x in words]
        
    def get_story_array(self, words, context_words):
        """get word ids for each phrase in a stimulus story
        """
        nctx = context_words + 1
        story_ids = self.encode(words)
        story_array = np.zeros([len(story_ids), nctx]) + self.UNK_ID
        for i in range(len(story_array)):
            segment = story_ids[i:i+nctx]
            story_array[i, :len(segment)] = segment
        return torch.tensor(story_array).long()

    def get_context_array(self, contexts):
        """get word ids for each context
        """
        context_array = np.array([self.encode(words) for words in contexts])
        return torch.tensor(context_array).long()

    def get_hidden(self, ids, layer):
        """get hidden layer representations
        """
        mask = torch.ones(ids.shape).int()
        with torch.no_grad():
            outputs = self.model(input_ids = ids.to(self.device), 
                                 attention_mask = mask.to(self.device), output_hidden_states = True)
        return outputs.hidden_states[layer].detach().cpu().numpy()

    def get_probs(self, ids):
        """get next word probability distributions
        """
        mask = torch.ones(ids.shape).int()
        with torch.no_grad():
            outputs = self.model(input_ids = ids.to(self.device), attention_mask = mask.to(self.device))
        probs = softmax(outputs.logits, dim = 2).detach().cpu().numpy()
        return probs