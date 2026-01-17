import torch
from torch.utils.data import DataLoader
from dataset import SkeletonDataset
from model import GaitModel

SEQ_LEN = 100
BATCH_SIZE = 8
CKPT = "checkpoints/gait_model.pth"
DATA = "datasets/skeleton_train_norm/train"

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = SkeletonDataset(DATA, seq_len=SEQ_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

sample_x, _, _ = dataset[0]
num_joints = sample_x.shape[1]
num_people = len(dataset.person_ids)

model = GaitModel(num_joints, num_people).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for x, pid, _ in loader:
        x, pid = x.to(device), pid.to(device)
        _, logits, _ = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == pid).sum().item()
        total += pid.size(0)

print(f"Accuracy: {100 * correct / total:.2f}%")
