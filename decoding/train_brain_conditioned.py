import os
import numpy as np
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import config
from BrainConditionedGPT import BrainConditionedGPT
from brain_data_utils import load_pca, build_training_data

np.random.seed(42)
torch.manual_seed(42)


class BrainTextDataset(Dataset):
    """Dataset of (text_context, brain_context, label) tuples."""

    def __init__(self, contexts, brain_contexts, word2id, unk_id,
                 noise_std=0.0):
        self.contexts = contexts
        self.brain_contexts = brain_contexts
        self.word2id = word2id
        self.unk_id = unk_id
        self.noise_std = noise_std

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        # Encode text context
        words = self.contexts[idx]
        ids = [self.word2id.get(w, self.unk_id) for w in words]
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)  # context (no target)
        label = ids[-1]  # next word target

        # Brain context
        bc = self.brain_contexts[idx].copy()  # (W, pca_dim)

        # Data augmentation
        if self.noise_std > 0:
            bc = bc + np.random.randn(*bc.shape).astype(np.float32) * self.noise_std

        bc = torch.tensor(bc, dtype=torch.float32)
        return input_ids, bc, label


def collate_fn(batch):
    # join variable-length sequences
    input_ids, brain_contexts, labels = zip(*batch)
    input_ids = torch.stack(input_ids)           # (B, context_words)
    brain_contexts = torch.stack(brain_contexts)  # (B, W, pca_dim)
    labels = torch.tensor(labels, dtype=torch.long)  # (B,)
    return input_ids, brain_contexts, labels


def evaluate(model, dataloader, criterion, device):
    model.eval_mode()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for input_ids, brain_contexts, labels in dataloader:
            # Get logits from brain-conditioned forward pass
            logits = model.get_probs_train(input_ids, brain_contexts)
            # Take logits at last position
            last_logits = logits[:, -1, :]  # (B, V)
            labels = labels.to(device)

            # get average loss
            loss = criterion(last_logits, labels)
            total_loss += loss.item() * len(labels)

            preds = last_logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += len(labels)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--sessions", nargs="+", type=int,
                        default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    parser.add_argument("--pca_path", type=str, default=None,
                        help="Path to PCA .npz file")
    parser.add_argument("--pca_dim", type=int, default=256)
    parser.add_argument("--use_em_voxels", action="store_true",
                        help="Use encoding model voxels only")
    parser.add_argument("--em_path", type=str, default=None,
                        help="Path to encoding model .npz (for voxel selection)")
    parser.add_argument("--encoder_layers", type=int, default=1)
    parser.add_argument("--encoder_ff_mult", type=int, default=2)
    parser.add_argument("--brain_window", type=int, default=20,
                        help="Number of TRs of brain context (default 20 = ~40s)")
    parser.add_argument("--cross_attn_layers", nargs="+", type=int, default=[11])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--noise_std", type=float, default=0.1,
                        help="Gaussian noise std for fMRI augmentation")
    parser.add_argument("--gate_reg", type=float, default=0.0,
                        help="L1 regularisation on gate values (keeps gates small)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--val_stories", type=int, default=4,
                        help="Number of stories held out for validation")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--pretrained_encoder", type=str, default=None,
                        help="Path to contrastive-pretrained brain encoder checkpoint")
    parser.add_argument("--freeze_ff", action="store_true",
                        help="Freeze FF paths in cross-attention (only cross-attn learns)")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze brain encoder (use with --pretrained_encoder)")
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()

    device = config.GPT_DEVICE
    print("Brain-Conditioned GPT Training")
    print(f"Subject: {args.subject}")
    print(f"Device: {device}")
    print(f"Cross-attn layers: {args.cross_attn_layers}")
    print(f"PCA dim: {args.pca_dim}")
    print(f"Brain window: {args.brain_window} TRs")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")

    # Load training stories
    stories = []
    with open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json"), "r") as f:
        sess_to_story = json.load(f)
    for sess in args.sessions:
        stories.extend(sess_to_story[str(sess)])
    print(f"Total stories: {len(stories)}")

    # Split train/val
    np.random.shuffle(stories)
    val_stories = stories[:args.val_stories]
    train_stories = stories[args.val_stories:]
    print(f"Train: {len(train_stories)} stories, Val: {len(val_stories)} stories")

    # Load PCA and optional voxel mask
    em_voxels = None
    if args.use_em_voxels:
        em_path = args.em_path or os.path.join(config.MODEL_DIR, args.subject, "encoding_model_perceived.npz")
        em_voxels = np.load(em_path)['voxels']
        print(f"Using encoding model voxels: {len(em_voxels)}")
    pca_path = args.pca_path
    if pca_path is None:
        pca_path = os.path.join(config.MODEL_DIR, args.subject,
                                f"pca_{args.pca_dim}.npz")
    print(f"Loading PCA from {pca_path}")
    pca_data = load_pca(pca_path)
    print(f"PCA: {pca_data['components'].shape[0]} components, "
          f"explains {pca_data['explained_variance_ratio'].sum():.1%} variance")

    # Load model
    print("Loading GPT model...")
    with open(os.path.join(config.DATA_LM_DIR, "perceived", "vocab.json"), "r") as f:
        vocab = json.load(f)

    model = BrainConditionedGPT(
        path=os.path.join(config.DATA_LM_DIR, "perceived", "model"),
        vocab=vocab, device=device, pca_dim=args.pca_dim,
        cross_attn_layers=tuple(args.cross_attn_layers), dropout=args.dropout,
        encoder_layers=args.encoder_layers, encoder_ff_mult=args.encoder_ff_mult
    )
    if args.pretrained_encoder:
        ckpt = torch.load(args.pretrained_encoder, map_location=device)
        model.brain_encoder.load_state_dict(ckpt['brain_encoder'])
        print(f"Loaded pretrained brain encoder from {args.pretrained_encoder}")
        if 'val_r1' in ckpt:
            print(f"  Pretrained R@1: {ckpt['val_r1']:.3f}, val loss: {ckpt['val_loss']:.4f}")
    if args.freeze_encoder:
        for p in model.brain_encoder.parameters():
            p.requires_grad = False
        print("Brain encoder frozen")
    if args.freeze_ff:
        model.freeze_ff_gates()
        print("FF paths frozen")
    n_params = sum(p.numel() for p in model.trainable_parameters())
    print(f"Trainable parameters: {n_params:,}")

    # Build training data
    print("\n\nBuilding training data...")
    train_data = build_training_data(args.subject, train_stories, pca_data,
                                     context_words=config.GPT_WORDS,
                                     brain_window=args.brain_window,
                                     voxel_mask=em_voxels)
    print(f"Training samples: {len(train_data['contexts'])}")

    print("\nBuilding validation data...")
    val_data = build_training_data(args.subject, val_stories, pca_data,
                                   context_words=config.GPT_WORDS,
                                   brain_window=args.brain_window,
                                   voxel_mask=em_voxels)
    print(f"Validation samples: {len(val_data['contexts'])}")

    # Create datasets
    train_dataset = BrainTextDataset(
        train_data['contexts'], train_data['brain_contexts'],
        model.word2id, model.UNK_ID, noise_std=args.noise_std
    )
    val_dataset = BrainTextDataset(
        val_data['contexts'], val_data['brain_contexts'],
        model.word2id, model.UNK_ID, noise_std=0.0
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn, num_workers=2)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # learning rate slowly decreases over time following a cosine curve
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.CrossEntropyLoss()

    # Save directory
    save_dir = os.path.join(config.MODEL_DIR, args.subject, "brain_conditioned" + args.suffix)
    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    best_val_loss = float('inf')

    # epochs since last improved
    patience_counter = 0

    print("Starting training")

    for epoch in range(1, args.epochs + 1):
        model.train_mode()
        train_loss = 0
        train_correct = 0
        train_total = 0

        # iterate through batches
        for batch_idx, (input_ids, brain_contexts, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward
            logits = model.get_probs_train(input_ids, brain_contexts)
            last_logits = logits[:, -1, :]  # (B, V) — predict next word
            labels = labels.to(device)

            loss = criterion(last_logits, labels)

            # Gate regularisation
            if args.gate_reg > 0:
                for layer_idx in model.cross_attn_layers:
                    m = model.cross_attn_modules[str(layer_idx)]
                    loss = loss + args.gate_reg * torch.abs(torch.tanh(m.gate_cross))

            loss.backward()

            # prevent exploding gradients and update weights
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(labels)
            preds = last_logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)

        scheduler.step()

        avg_train_loss = train_loss / train_total if train_total > 0 else float('inf')
        train_acc = train_correct / train_total if train_total > 0 else 0

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Monitor log gate values
        gate_values = {}
        for layer_idx in model.cross_attn_layers:
            m = model.cross_attn_modules[str(layer_idx)]
            gc = torch.tanh(m.gate_cross).item()
            gf = torch.tanh(m.gate_ff).item()
            gate_values[layer_idx] = (gc, gf)

        gate_parts = []
        for layer, (gc, gf) in gate_values.items():
            gate_parts.append(f"Layer {layer}: cross={gc:.3f} ff={gf:.3f}")
        gate_str = " | ".join(gate_parts)

        print(f"Epoch {epoch:3d} | "
              f"Train loss: {avg_train_loss:.4f} acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f} acc: {val_acc:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | "
              f"Gates: {gate_str}")

        # Save checkpoint
        if epoch % args.save_every == 0:
            model.save_trainable(os.path.join(save_dir, f"epoch_{epoch}.pt"))

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save_trainable(os.path.join(save_dir, "best.pt"))
            print(f"New best val loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience {args.patience})")
                break

    # Save final
    model.save_trainable(os.path.join(save_dir, "last.pt"))
    print(f"Training complete")
    print(f"Best val loss: {best_val_loss:.4f}")
