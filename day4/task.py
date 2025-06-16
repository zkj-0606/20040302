import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from einops import rearrange, repeat

from DAY3.dataset import ImageTxtDataset  # 保持原始自定义数据集路径

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
##########
# ------------------------- Vision Transformer 结构 -------------------------
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, heads, dim_head, dropout),
                FeedForward(dim, mlp_dim, dropout)
            ]) for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, seq_len, patch_size, num_classes, dim, depth,
                 heads, mlp_dim, channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert seq_len % patch_size == 0, "序列长度必须能被 patch_size 整除"
        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size

        self.to_patch_embedding = nn.Sequential(
            rearrange('b c (n p) -> b n (p c)', p=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :x.shape[1]]
        x = self.dropout(x)
        x = self.transformer(x)
        return self.mlp_head(x[:, 0])

# ------------------------- 配置与预处理 -------------------------

class Config:
    train_txt = r"D:\intership\DAY3\data_set\train.txt"
    train_img_dir = r"D:\intership\DAY3\data_set\Images\train"
    val_txt   = r"D:\intership\DAY3\data_set\val.txt"
    val_img_dir   = r"D:\intership\DAY3\data_set\Images\val"
    batch_size = 64
    lr = 1e-4
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = "logs_vit_rewrite"
    model_dir = "model_save"

transform = transforms.Compose([
    transforms.Resize((1, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = ImageTxtDataset(Config.train_txt, Config.train_img_dir, transform)
val_dataset   = ImageTxtDataset(Config.val_txt,   Config.val_img_dir,   transform)

train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=Config.batch_size)

num_classes = max(train_dataset.labels) + 1
print(f"标签范围: {min(train_dataset.labels)} ~ {max(train_dataset.labels)}")
print(f"自动设置 num_classes = {num_classes}")

# ------------------------- 模型、损失函数、优化器 -------------------------

model = ViT(
    seq_len=256,
    patch_size=16,
    num_classes=num_classes,
    dim=1024,
    depth=6,
    heads=8,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
).to(Config.device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
writer = SummaryWriter(Config.log_dir)

# ------------------------- 训练与验证函数 -------------------------

def train_one_epoch(epoch, step):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(Config.device), labels.to(Config.device)
        imgs = imgs.squeeze(2)  # (B, C, 1, 256) → (B, C, 256)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"[Epoch {epoch+1}] Step {step}: Loss = {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), step)
        step += 1
    return step

def evaluate(epoch):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(Config.device), labels.to(Config.device)
            imgs = imgs.squeeze(2)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    acc = correct / len(val_dataset)
    print(f"[Eval] Loss = {avg_loss:.4f}, Accuracy = {acc:.4f}")
    writer.add_scalar("test_loss", avg_loss, epoch)
    writer.add_scalar("test_accuracy", acc, epoch)

# ------------------------- 主训练循环 -------------------------

total_step = 0
start_time = time.time()
for epoch in range(Config.epochs):
    print(f"\n===== 第 {epoch+1} 轮训练开始 =====")
    total_step = train_one_epoch(epoch, total_step)
    evaluate(epoch)
    os.makedirs(Config.model_dir, exist_ok=True)
    model_path = os.path.join(Config.model_dir, f"vit_epoch{epoch+1}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存: {model_path}")

writer.close()
print(f"训练完成，总耗时: {(time.time() - start_time):.2f}s")