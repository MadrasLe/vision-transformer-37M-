import math, os, random, time, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# -----------------------
# Utils
# -----------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -----------------------
# Vision Transformer
# -----------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=256):
        super().__init__()
        assert img_size % patch_size == 0, "img_size deve ser múltiplo de patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid[0] * self.grid[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                          # (B, D, Gh, Gw)
        x = x.flatten(2).transpose(1, 2)          # (B, N, D)
        return x

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim deve ser múltiplo de num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, drop=drop_rate, attn_drop=drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head(cls_out)

# -----------------------
# Dados
# -----------------------
def build_transforms(img_size):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
    ])
    return train_tf, val_tf

def build_datasets(args):
    train_tf, val_tf = build_transforms(args.img_size)
    if args.dataset.lower() == "cifar10":
        train_ds = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=train_tf)
        val_ds   = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=val_tf)
        num_classes = 10
    elif args.dataset.lower() == "cifar100":
        train_ds = datasets.CIFAR100(root=args.data_root, train=True, download=True, transform=train_tf)
        val_ds   = datasets.CIFAR100(root=args.data_root, train=False, download=True, transform=val_tf)
        num_classes = 100
    elif args.dataset.lower() == "imagefolder":
        train_dir = os.path.join(args.data_root, "train")
        val_dir   = os.path.join(args.data_root, "val")
        train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
        val_ds   = datasets.ImageFolder(val_dir, transform=val_tf)
        num_classes = len(train_ds.classes)
    else:
        raise ValueError("dataset deve ser 'cifar10', 'cifar100' ou 'imagefolder'")
    return train_ds, val_ds, num_classes

# -----------------------
# Opt e Scheduler
# -----------------------
def build_optimizer(model, lr, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith("bias") or "pos_embed" in n or "cls_token" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    optim_groups = [
        {"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)

class CosineWarmup:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self._step = 0
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]

    def step(self):
        self._step += 1
        for i, group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            if self._step <= self.warmup_steps:
                lr = base_lr * self._step / max(1, self.warmup_steps)
            else:
                t = (self._step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                cos = 0.5 * (1 + math.cos(math.pi * t))
                lr = self.min_lr + (base_lr - self.min_lr) * cos
            group['lr'] = lr

# -----------------------
# Loop treino/val
# -----------------------
def train_one_epoch(args, model, loader, optimizer, scaler, device, scheduler=None):
    model.train()
    running_loss, running_acc = 0.0, 0.0
    steps = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=args.amp):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        running_acc  += accuracy(logits.detach(), labels)
        steps += 1

    return running_loss / steps, running_acc / steps

@torch.no_grad()
def evaluate(args, model, loader, device):
    model.eval()
    running_loss, running_acc = 0.0, 0.0
    steps = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.cuda.amp.autocast(enabled=args.amp):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        running_loss += loss.item()
        running_acc  += accuracy(logits, labels)
        steps += 1
    return running_loss / steps, running_acc / steps

def save_ckpt(path, model, optimizer, epoch, best_acc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
    }, path)

def load_ckpt(path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("best_acc", 0.0)

# -----------------------
# Main
# -----------------------
def main():
    p = argparse.ArgumentParser(description="Treino de Vision Transformer (pequeno)")
    p.add_argument("--data-root", type=str, default="./data", help="raiz dos dados")
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "imagefolder"])
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--patch", type=int, default=4)
    p.add_argument("--model-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--drop", type=float, default=0.1)
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--amp", action="store_true", help="ativa mixed precision")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--outdir", type=str, default="runs/vit_small")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--preset", type=str, default=None, choices=[None,"smallpp"])
    args = p.parse_args()

    # Preset para ViT Small++
    if args.preset == "smallpp":
        args.model_dim = 384
        args.depth = 8
        args.heads = 12
        args.mlp_ratio = 4
        args.batch_size = 256
        args.epochs = 100
        if args.dataset == "cifar100" and args.num_classes is None:
            args.num_classes = 100

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, inferred_classes = build_datasets(args)
    num_classes = args.num_classes if args.num_classes is not None else inferred_classes

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = ViT(
        img_size=args.img_size,
        patch_size=args.patch,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=args.model_dim,
        depth=args.depth,
        num_heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        drop_rate=args.drop,
    ).to(device)

    print(f"📦 Parâmetros treináveis: {count_params(model)/1e6:.2f}M")

    optimizer = build_optimizer(model, lr=args.lr, wd=args.wd)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(1, args.warmup_epochs * len(train_loader))
    scheduler = CosineWarmup(optimizer, warmup_steps, total_steps, min_lr=args.min_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch, best_acc = 0, 0.0
    if args.ckpt and args.resume:
        if os.path.exists(args.ckpt):
            start_epoch, best_acc = load_ckpt(args.ckpt, model, optimizer, map_location=device)
            print(f"🔁 Resumido de {args.ckpt} (epoch={start_epoch}, best_acc={best_acc:.4f})")
        else:
            print(f"⚠️  Checkpoint não encontrado em {args.ckpt} — iniciando do zero.")

    if args.eval:
        assert args.ckpt is not None and os.path.exists(args.ckpt), "--eval exige --ckpt válido"
        _e, _b = load_ckpt(args.ckpt, model, optimizer=None, map_location=device)
        val_loss, val_acc = evaluate(args, model, val_loader, device)
        print(f"[EVAL] loss={val_loss:.4f} | acc={val_acc*100:.2f}%")
        return

    os.makedirs(args.outdir, exist_ok=True)
    best_path = os.path.join(args.outdir, "best.pt")
    last_path = os.path.join(args.outdir, "last.pt")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(args, model, train_loader, optimizer, scaler, device, scheduler)
        va_loss, va_acc = evaluate(args, model, val_loader, device)

        is_best = va_acc > best_acc
        best_acc = max(best_acc, va_acc)
        save_ckpt(last_path, model, optimizer, epoch+1, best_acc)
        if is_best:
            save_ckpt(best_path, model, optimizer, epoch+1, best_acc)

        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"train: loss {tr_loss:.4f} acc {tr_acc*100:.2f}% | "
              f"val: loss {va_loss:.4f} acc {va_acc*100:.2f}% | "
              f"best {best_acc*100:.2f}% | {dt:.1f}s")

if __name__ == "__main__":
    main()
