"""Stage 1: Contrastive pre-training for BrainEncoder.

Trains the BrainEncoder to align PCA-projected fMRI windows with
GPT-1 layer-9 text embeddings via CLIP-style InfoNCE loss.
"""
import os
import numpy as np
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import config
from GPT import GPT
from BrainConditionedGPT import BrainEncoder
from StimulusModel import LMFeatures
from brain_data_utils import load_pca, build_contrastive_data

np.random.seed(42)
torch.manual_seed(42)


class ContrastiveHead(nn.Module):
    """Projection head for contrastive training. Discarded at Stage 2."""

    def __init__(self, dim=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        """x: (B, W, D) -> mean pool -> project -> L2 normalize -> (B, D)"""
        x = x.mean(dim=1)
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return x


class ContrastiveDataset(Dataset):
    def __init__(self, brain_windows, text_targets, noise_std=0.0):
        self.brain_windows = brain_windows
        self.text_targets = text_targets
        self.noise_std = noise_std

    def __len__(self):
        return len(self.brain_windows)

    def __getitem__(self, idx):
        brain = self.brain_windows[idx].copy()
        if self.noise_std > 0:
            brain = brain + np.random.randn(*brain.shape).astype(np.float32) * self.noise_std
        text = self.text_targets[idx]
        return torch.tensor(brain, dtype=torch.float32), torch.tensor(text, dtype=torch.float32)


def clip_loss(brain_emb, text_emb, log_temperature):
    """Symmetric CLIP-style InfoNCE loss.

    Args:
        brain_emb: (B, D) L2-normalized brain embeddings
        text_emb: (B, D) L2-normalized text embeddings
        log_temperature: learnable log(temperature) scalar

    Returns:
        loss, temperature value
    """
    temperature = torch.clamp(log_temperature.exp(), min=0.01, max=1.0)
    logits = brain_emb @ text_emb.T / temperature  # (B, B)
    labels = torch.arange(len(brain_emb), device=brain_emb.device)
    loss_b2t = F.cross_entropy(logits, labels)
    loss_t2b = F.cross_entropy(logits.T, labels)
    return (loss_b2t + loss_t2b) / 2, temperature.item()


def compute_retrieval(brain_emb, text_emb, ks=(1, 5)):
    """Compute retrieval R@k metrics.

    Args:
        brain_emb: (N, D) L2-normalized
        text_emb: (N, D) L2-normalized
        ks: tuple of k values

    Returns:
        dict of {f'R@{k}': accuracy}
    """
    sim = brain_emb @ text_emb.T  # (N, N)
    results = {}
    for k in ks:
        topk = sim.topk(k, dim=1).indices  # (N, k)
        labels = torch.arange(len(brain_emb), device=brain_emb.device).unsqueeze(1)
        correct = (topk == labels).any(dim=1).float().mean().item()
        results[f'R@{k}'] = correct
    return results


@torch.no_grad()
def evaluate(encoder, head, dataloader, log_temperature, device):
    """Evaluate contrastive model."""
    encoder.eval()
    head.eval()

    all_brain_emb = []
    all_text_emb = []
    total_loss = 0
    n_batches = 0

    for brain_batch, text_batch in dataloader:
        brain_batch = brain_batch.to(device)
        text_batch = text_batch.to(device)

        brain_tokens = encoder(brain_batch)
        brain_emb = head(brain_tokens)
        text_emb = F.normalize(text_batch, dim=-1)

        loss, _ = clip_loss(brain_emb, text_emb, log_temperature)
        total_loss += loss.item()
        n_batches += 1

        all_brain_emb.append(brain_emb.cpu())
        all_text_emb.append(text_emb.cpu())

    all_brain_emb = torch.cat(all_brain_emb)
    all_text_emb = torch.cat(all_text_emb)
    retrieval = compute_retrieval(all_brain_emb, all_text_emb, ks=(1, 5))

    return total_loss / n_batches, retrieval


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--sessions", nargs="+", type=int,
                        default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    parser.add_argument("--pca_dim", type=int, default=256)
    parser.add_argument("--use_em_voxels", action="store_true",
                        help="Use only encoding model voxels (top 10k language-responsive)")
    parser.add_argument("--em_path", type=str, default=None,
                        help="Path to encoding model .npz (for voxel selection)")
    parser.add_argument("--encoder_layers", type=int, default=1)
    parser.add_argument("--encoder_ff_mult", type=int, default=2)
    parser.add_argument("--brain_window", type=int, default=20)
    parser.add_argument("--hop_length", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--noise_std", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val_stories", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()

    device = config.GPT_DEVICE
    print("=" * 60)
    print("Contrastive Pre-Training: BrainEncoder")
    print("=" * 60)
    print(f"Subject: {args.subject}, Device: {device}")
    print(f"Brain window: {args.brain_window} TRs, Hop: {args.hop_length}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")

    # Load stories
    stories = []
    with open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json"), "r") as f:
        sess_to_story = json.load(f)
    for sess in args.sessions:
        stories.extend(sess_to_story[str(sess)])
    print(f"Total stories: {len(stories)}")

    np.random.shuffle(stories)
    val_stories = stories[:args.val_stories]
    train_stories = stories[args.val_stories:]
    print(f"Train: {len(train_stories)}, Val: {len(val_stories)}")

    # Load PCA (optionally refit on encoding model voxels only)
    em_voxels = None
    if args.use_em_voxels:
        em_path = args.em_path or os.path.join(config.MODEL_DIR, args.subject, "encoding_model_perceived.npz")
        em_voxels = np.load(em_path)['voxels']
        print(f"Using encoding model voxels: {len(em_voxels)} language-responsive voxels")
        # Refit PCA on just these voxels
        from brain_data_utils import fit_pca
        pca_save = os.path.join(config.MODEL_DIR, args.subject,
                                f"pca_{args.pca_dim}_emvox.npz")
        if os.path.exists(pca_save):
            pca_data = load_pca(pca_save)
            print(f"Loaded cached EM-voxel PCA: {pca_data['components'].shape[0]} components, "
                  f"explains {pca_data['explained_variance_ratio'].sum():.1%}")
        else:
            fit_pca(args.subject, train_stories + val_stories,
                    n_components=args.pca_dim, save_path=pca_save,
                    voxel_mask=em_voxels)
            pca_data = load_pca(pca_save)
    else:
        pca_path = os.path.join(config.MODEL_DIR, args.subject, f"pca_{args.pca_dim}.npz")
        pca_data = load_pca(pca_path)
        print(f"PCA: {pca_data['components'].shape[0]} components")

    # Load GPT-1 for text embeddings
    print("Loading GPT-1 for text embeddings...")
    with open(os.path.join(config.DATA_LM_DIR, "perceived", "vocab.json"), "r") as f:
        gpt_vocab = json.load(f)
    gpt = GPT(path=os.path.join(config.DATA_LM_DIR, "perceived", "model"),
              vocab=gpt_vocab, device=device)
    features = LMFeatures(model=gpt, layer=config.GPT_LAYER, context_words=config.GPT_WORDS)

    # Build contrastive pairs
    print("\nBuilding training pairs...")
    train_brain, train_text = build_contrastive_data(
        args.subject, train_stories, pca_data, features,
        brain_window=args.brain_window, hop_length=args.hop_length,
        voxel_mask=em_voxels
    )
    print("Building validation pairs...")
    val_brain, val_text = build_contrastive_data(
        args.subject, val_stories, pca_data, features,
        brain_window=args.brain_window, hop_length=args.hop_length,
        voxel_mask=em_voxels
    )

    # Free GPT memory
    del gpt, features
    torch.cuda.empty_cache()

    # Datasets
    train_dataset = ContrastiveDataset(train_brain, train_text, noise_std=args.noise_std)
    val_dataset = ContrastiveDataset(val_brain, val_text, noise_std=0.0)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    encoder = BrainEncoder(pca_dim=args.pca_dim, hidden_dim=768, dropout=0.2,
                           n_layers=args.encoder_layers, ff_mult=args.encoder_ff_mult).to(device)
    head = ContrastiveHead(dim=768).to(device)
    log_temperature = nn.Parameter(torch.tensor(0.07, device=device).log())

    n_encoder = sum(p.numel() for p in encoder.parameters())
    n_head = sum(p.numel() for p in head.parameters())
    print(f"Encoder params: {n_encoder:,}, Head params: {n_head:,}, Total: {n_encoder + n_head:,}")

    # Optimizer
    optimizer = torch.optim.AdamW([
        {'params': encoder.parameters()},
        {'params': head.parameters()},
        {'params': [log_temperature], 'weight_decay': 0.0},
    ], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Save dir
    save_dir = os.path.join(config.MODEL_DIR, args.subject,
                            "brain_conditioned" + args.suffix)
    os.makedirs(save_dir, exist_ok=True)

    # Training
    best_val_loss = float('inf')
    patience_counter = 0

    print("\n" + "=" * 60)
    print("Starting contrastive training...")
    print(f"Random baseline loss: ~{np.log(args.batch_size):.2f}")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        head.train()
        train_loss = 0
        n_batches = 0

        for brain_batch, text_batch in train_loader:
            brain_batch = brain_batch.to(device)
            text_batch = text_batch.to(device)

            optimizer.zero_grad()

            brain_tokens = encoder(brain_batch)
            brain_emb = head(brain_tokens)
            text_emb = F.normalize(text_batch, dim=-1)

            loss, temp = clip_loss(brain_emb, text_emb, log_temperature)

            if torch.isnan(loss):
                print(f"WARNING: NaN at epoch {epoch}")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(head.parameters()) + [log_temperature],
                max_norm=1.0
            )
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = train_loss / max(n_batches, 1)

        # Validate
        val_loss, retrieval = evaluate(encoder, head, val_loader, log_temperature, device)

        print(f"Epoch {epoch:3d} | "
              f"Train: {avg_train:.4f} | Val: {val_loss:.4f} | "
              f"Temp: {temp:.4f} | "
              f"R@1: {retrieval['R@1']:.3f} R@5: {retrieval['R@5']:.3f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                'brain_encoder': encoder.state_dict(),
                'head': head.state_dict(),
                'log_temperature': log_temperature.data,
                'epoch': epoch,
            }, os.path.join(save_dir, f"contrastive_epoch_{epoch}.pt"))

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'brain_encoder': encoder.state_dict(),
                'head': head.state_dict(),
                'log_temperature': log_temperature.data,
                'epoch': epoch,
                'val_loss': val_loss,
                'val_r1': retrieval['R@1'],
                'val_r5': retrieval['R@5'],
            }, os.path.join(save_dir, "best.pt"))
            print(f"  -> New best val loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final save
    torch.save({
        'brain_encoder': encoder.state_dict(),
        'epoch': epoch,
    }, os.path.join(save_dir, "contrastive_last.pt"))

    print("\n" + "=" * 60)
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Models saved in: {save_dir}")
    print("=" * 60)
