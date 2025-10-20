import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
from pathlib import Path

# -------------------------------------------------------
# 1) KD loss
# -------------------------------------------------------
class DistillLoss(nn.Module):
    def __init__(self, T=2.0, alpha=0.7):
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        loss_ce = self.ce(student_logits, labels)
        # KL between softened distributions
        p_teacher = F.log_softmax(teacher_logits / self.T, dim=1)
        p_student = F.log_softmax(student_logits / self.T, dim=1)
        loss_kd = F.kl_div(p_student, p_teacher.exp(), reduction="batchmean") * (self.T**2)
        return self.alpha * loss_ce + (1 - self.alpha) * loss_kd

# -------------------------------------------------------
# 2) Student network (choose MobileNetV3-Small)
# -------------------------------------------------------
def build_student(num_classes):
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model

# -------------------------------------------------------
# 3) Teacher inference wrapper (ONNX)
# -------------------------------------------------------
class ONNXTeacher:
    def __init__(self, model_paths):
        self.sessions = [ort.InferenceSession(str(p)) for p in model_paths]
        self.input_name = self.sessions[0].get_inputs()[0].name

    def predict_logits(self, x_np):
        # x_np: NCHW float32
        logits_all = []
        for sess in self.sessions:
            logits = sess.run(None, {self.input_name: x_np})[0]
            logits_all.append(logits)
        return np.mean(np.stack(logits_all, axis=0), axis=0)  # ensemble avg

# -------------------------------------------------------
# 4) Custom dataset (apply preprocessing.json rules)
# -------------------------------------------------------
class LCZ42Dataset(Dataset):
    def __init__(self, img_paths, labels, preprocess_json):
        self.img_paths = img_paths
        self.labels = labels
        with open(preprocess_json,'r') as f:
            self.meta = json.load(f)

        self.transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),  # 0–1
            # If z-score enabled, apply here
            transforms.Normalize(mean=self.meta["zscore"]["mu"],
                                 std=self.meta["zscore"]["sigma"])
              if self.meta["zscore"]["enabled"] else transforms.Lambda(lambda x: x)
        ])

    def __len__(self): return len(self.img_paths)
    def __getitem__(self, idx):
        from PIL import Image
        x = Image.open(self.img_paths[idx]).convert("RGB")  # assuming pre-made 3-band
        y = self.labels[idx]
        return self.transform(x), y

# -------------------------------------------------------
# 5) Training loop
# -------------------------------------------------------
def train_student(student, teacher, train_loader, val_loader, num_classes, epochs=20, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = student.to(device)
    optimizer = optim.Adam(student.parameters(), lr=lr)
    criterion = DistillLoss(T=2.0, alpha=0.7)

    for epoch in range(epochs):
        student.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # Teacher logits
            logits_teacher = teacher.predict_logits(x.cpu().numpy())
            logits_teacher = torch.tensor(logits_teacher, device=device)
            # Forward student
            logits_student = student(x)
            loss = criterion(logits_student, logits_teacher, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss {total_loss/len(train_loader):.4f}")

        # Validation accuracy
        student.eval(); correct=0; total=0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                out = student(x)
                pred = out.argmax(1)
                correct += (pred==y).sum().item()
                total += y.size(0)
        print(f" Val Acc {correct/total:.3f}")

    return student

# -------------------------------------------------------
# 6) Save student to ONNX
# -------------------------------------------------------
def export_student(student, out_path, num_classes):
    dummy = torch.randn(1,3,32,32)
    torch.onnx.export(student, dummy, out_path,
                      input_names=['input'], output_names=['logits'],
                      opset_version=17, dynamic_axes={'input':{0:'batch'}})
    print(f"[+] Exported student to {out_path}")

# -------------------------------------------------------
# Main example usage
# -------------------------------------------------------
if __name__ == "__main__":
    # paths
    onnx_models = list(Path("deployment/onnx").glob("dense_*.onnx"))
    teacher = ONNXTeacher(onnx_models)

    # Load your dataset splits (here, placeholders)
    train_paths, train_labels = [...], [...]
    val_paths, val_labels     = [...], [...]
    preprocess_json = "deployment/onnx/preprocessing.json"

    train_ds = LCZ42Dataset(train_paths, train_labels, preprocess_json)
    val_ds   = LCZ42Dataset(val_paths, val_labels, preprocess_json)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=64)

    student = build_student(num_classes=len(train_ds.meta["labels"]))
    student = train_student(student, teacher, train_loader, val_loader,
                            num_classes=len(train_ds.meta["labels"]),
                            epochs=20, lr=1e-3)

    export_student(student, "deployment/student_mobilenetv3.onnx",
                   num_classes=len(train_ds.meta["labels"]))