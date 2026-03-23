import math
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM
from torch.nn.functional import softmax


class BrainEncoder(nn.Module):
    # takes PCA-projected fMRI and converts into "brain tokens" which cross-attention layers
    # can attend to
    def __init__(self, pca_dim=256, hidden_dim=768, n_heads=8, n_layers=2,
                 ff_mult=4, dropout=0.2):
        super().__init__()
        # single linear layer, each 256 vector for each TR is mapped to 768 embedding/tokem
        self.proj = nn.Linear(pca_dim, hidden_dim)

        # learnable positional embeddings so model can tell which TR is which
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, hidden_dim))  # max 64 TRs
        nn.init.normal_(self.pos_embed, std=0.02)

        # 2-layer transformer encoder
        # each layer does: self-attention (8 heads) to feed-forward (768 -> 3072 -> 768)
        # lets TRs attend to each other 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * ff_mult,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # (B, W, pca_dim) -> (B, W, hidden_dim)
        B, W, _ = x.shape
        # project x from 256 to 768 and add positional embeddings
        x = self.proj(x) + self.pos_embed[:, :W, :]
        # run through 2 self-attention layers
        x = self.transformer(x)
        x = self.norm(x)
        # output has turned 20 TRs of 256 dim each to 20 "brain tokens"
        return x


class GatedCrossAttentionLayer(nn.Module):
    #Flamingo-style gated cross-attention + feed-forward.
    def __init__(self, hidden_dim=768, n_heads=12, ff_dim=3072, dropout=0.2):
        super().__init__()
        #pre-norm
        self.ln_cross = nn.LayerNorm(hidden_dim)

        # cross attention
        # Q comes from text hidden states
        # K and V come from brain tokens/brain encoder output
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )

        # learnable param that controls how much the cross-attention output gets mixed in
        self.gate_cross = nn.Parameter(torch.zeros(1))  # tanh(0) = 0

        # ff layer lets model further process result of cross attention
        self.ln_ff = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.gate_ff = nn.Parameter(torch.zeros(1))  # tanh(0) = 0

    def forward(self, x, brain_tokens):
        # x / text hidden states: (B, T, D)
        # brain_tokens: (B, W, D)
        # returns: (B, T, D)

        # Gated cross-attention
        residual = x
        x_norm = self.ln_cross(x)

        # cross attention each text position calcs attention weights over 20 brain tokens and takes
        # weighted sum
        attn_out, _ = self.cross_attn(query=x_norm, key=brain_tokens, value=brain_tokens)
        # mixes in attention output to residual
        x = residual + torch.tanh(self.gate_cross) * attn_out

        # Gated feed-forward
        residual = x
        x = residual + torch.tanh(self.gate_ff) * self.ff(self.ln_ff(x))

        return x


class BrainConditionedGPT():
    # Wraps frozen GPT-1 and injects gated cross-attention at specified layers.
    # When brain_context=None behaves same as plain GPT.

    def __init__(self, path, vocab, device='cpu', pca_dim=256,
                 cross_attn_layers=(11,), dropout=0.2,
                 encoder_layers=1, encoder_ff_mult=2):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(path).eval().to(self.device)
        self.vocab = vocab
        self.word2id = {w: i for i, w in enumerate(self.vocab)}
        self.UNK_ID = self.word2id['<unk>']

        # freeze all GPT parameters so backprop doesnt update GPT params
        for param in self.model.parameters():
            param.requires_grad = False

        hidden_dim = self.model.config.n_embd  # 768
        n_heads = self.model.config.n_head      # 12

        # Brain encoder
        self.brain_encoder = BrainEncoder(
            pca_dim=pca_dim, hidden_dim=hidden_dim, dropout=dropout,
            n_layers=encoder_layers, ff_mult=encoder_ff_mult
        ).to(self.device)

        # Gated cross-attention layers inserted after specified GPT blocks
        self.cross_attn_layers = sorted(cross_attn_layers)
        self.cross_attn_modules = nn.ModuleDict()
        for layer_idx in self.cross_attn_layers:
            self.cross_attn_modules[str(layer_idx)] = GatedCrossAttentionLayer(
                hidden_dim=hidden_dim, n_heads=n_heads, dropout=dropout
            ).to(self.device)

    def freeze_ff_gates(self):
       # Freeze FF paths in cross-attention layers (set gate_ff=0, freeze FF params)
        for layer_idx in self.cross_attn_layers:
            m = self.cross_attn_modules[str(layer_idx)]
            m.gate_ff.requires_grad = False
            for p in m.ff.parameters():
                p.requires_grad = False
            for p in m.ln_ff.parameters():
                p.requires_grad = False

    def trainable_parameters(self):
        # collect list of params which optimizer should update
        params = list(self.brain_encoder.parameters())
        params += list(self.cross_attn_modules.parameters())
        return [p for p in params if p.requires_grad]

    def train_mode(self):
        # Set trainable components to train mode i.e. enable dropout
        self.brain_encoder.train()
        self.cross_attn_modules.train()

    def eval_mode(self):
        # Set all components to eval mode i.e. disable dropout
        self.brain_encoder.eval()
        self.cross_attn_modules.eval()

    def save_trainable(self, path):
        state = {
            'brain_encoder': self.brain_encoder.state_dict(),
            'cross_attn_modules': self.cross_attn_modules.state_dict(),
            'cross_attn_layers': self.cross_attn_layers,
        }
        torch.save(state, path)

    def load_trainable(self, path):
        state = torch.load(path, map_location=self.device)
        self.brain_encoder.load_state_dict(state['brain_encoder'])
        self.cross_attn_modules.load_state_dict(state['cross_attn_modules'])

    def encode(self, words):
        """Map from words to ids.
        """
        return [self.word2id[x] if x in self.word2id else self.UNK_ID for x in words]

    def get_story_array(self, words, context_words):
        """Get word ids for each phrase in a stimulus story.
        """
        nctx = context_words + 1
        story_ids = self.encode(words)
        story_array = np.zeros([len(story_ids), nctx]) + self.UNK_ID
        for i in range(len(story_array)):
            segment = story_ids[i:i+nctx]
            story_array[i, :len(segment)] = segment
        return torch.tensor(story_array).long()

    def get_context_array(self, contexts):
        """Get word ids for each context.
        """
        context_array = np.array([self.encode(words) for words in contexts])
        return torch.tensor(context_array).long()

    def _forward_with_brain(self, ids, brain_context=None, output_hidden_states=False):
        # Run GPT forward pass and put in cross-attention after specified layers
        # returns (logits, hidden_states) where hidden_states is None unless requested

        # If no brain context just use normal GPT
        if brain_context is None:
            mask = torch.ones(ids.shape).int()
            with torch.no_grad():
                outputs = self.model(
                    input_ids=ids.to(self.device),
                    attention_mask=mask.to(self.device),
                    output_hidden_states=output_hidden_states
                )
            if output_hidden_states:
                return outputs.logits, outputs.hidden_states
            return outputs.logits, None

        # Encode brain context
        brain_tokens = self.brain_encoder(brain_context.to(self.device))  # (B, W, 768)

        transformer = self.model.transformer
        input_shape = ids.size()
        ids = ids.to(self.device)

        # Embeddings
        position_ids = torch.arange(input_shape[1], device=self.device).unsqueeze(0)
        inputs_embeds = transformer.tokens_embed(ids)
        position_embeds = transformer.positions_embed(position_ids)
        hidden = inputs_embeds + position_embeds
        hidden = transformer.drop(hidden)

        hidden_states = [hidden] if output_hidden_states else None

        # Loop through transformer blocks
        for i, block in enumerate(transformer.h):
            hidden = block(hidden)[0]  # Block returns tuple, first element is hidden

            # Insert cross-attention after specified layers
            if i in self.cross_attn_layers:
                hidden = self.cross_attn_modules[str(i)](hidden, brain_tokens)

            if output_hidden_states:
                hidden_states.append(hidden)

        # put final hidden states through GPT to produce logits for next work pred
        logits = self.model.lm_head(hidden)

        return logits, hidden_states

    def get_hidden(self, ids, layer, brain_context=None):
        """Get hidden layer representations.
        """

        # Same as GPT with optional brain conditioning
        if brain_context is None:
            # Exact vanilla path
            mask = torch.ones(ids.shape).int()
            with torch.no_grad():
                outputs = self.model(
                    input_ids=ids.to(self.device),
                    attention_mask=mask.to(self.device),
                    output_hidden_states=True
                )
            return outputs.hidden_states[layer].detach().cpu().numpy()

        # Brain-conditioned path
        _, hidden_states = self._forward_with_brain(
            ids, brain_context=brain_context, output_hidden_states=True
        )
        return hidden_states[layer].detach().cpu().numpy()

    def get_probs(self, ids, brain_context=None):
        """Get next word probability distributions.
        """

        # Same as GPT with optional brain conditioning
        if brain_context is None:
            # Exact vanilla path
            mask = torch.ones(ids.shape).int()
            with torch.no_grad():
                outputs = self.model(
                    input_ids=ids.to(self.device),
                    attention_mask=mask.to(self.device)
                )
            probs = softmax(outputs.logits, dim=2).detach().cpu().numpy()
            return probs

        # Brain-conditioned path
        logits, _ = self._forward_with_brain(ids, brain_context=brain_context)
        probs = softmax(logits, dim=2).detach().cpu().numpy()
        return probs

    def get_probs_train(self, ids, brain_context):
        # Get next word logits for training and preserve gradients

        logits, _ = self._forward_with_brain(ids, brain_context=brain_context)
        return logits
