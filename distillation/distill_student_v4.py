import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import numpy as np
import json
from pathlib import Path

# optional: timm for MobileNetV4
try:
    import timm
    HAS_TIMM = True
except Exception:
    HAS_TIMM = False

# -------------------------------------------------------
# 1) KD loss (same as your file)
# -------------------------------------------------------
class DistillLoss(nn.Module):
    def __init__(self, T=2.0, alpha=0.7):
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        loss_ce = self.ce(student_logits, labels)
        pt = F.log_softmax(teacher_logits / self.T, dim=1)
        ps = F.log_softmax(student_logits / self.T, dim=1)
        loss_kd = F.kl_div(ps, pt.exp(), reduction="batchmean") * (self.T**2)
        return self.alpha * loss_ce + (1 - self.alpha) * loss_kd

# -------------------------------------------------------
# 2) Student = MobileNetV4-medium (via timm) @ 224x224
#    Fallback -> EfficientNet-B0 (torchvision)
# -------------------------------------------------------
def build_student(num_classes, arch="mobilenetv4m"):
    if arch == "mobilenetv4m":
        if not HAS_TIMM:
            print("[warn] timm not found -> falling back to EfficientNet-B0")
            return build_student(num_classes, arch="efficientnet_b0")
        # timm model names may vary; these are common variants:
        #   'mobilenetv4_conv_medium' or 'mobilenetv4_medium'
        name_candidates = [
            "mobilenetv4_conv_medium",
            "mobilenetv4_medium",
            "mobilenetv4_m"  # safety alias
        ]
        last_err = None
        for nm in name_candidates:
            try:
                model = timm.create_model(nm, pretrained=False, num_classes=num_classes, in_chans=3)
                return model
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"Could not instantiate MobileNetV4-medium via timm. Last error: {last_err}")
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    elif arch == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unknown arch: {arch}")

# -------------------------------------------------------
# 3) Teacher inference wrapper (ONNX, same logic)
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
        return np.mean(np.stack(logits_all, axis=0), axis=0)

# -------------------------------------------------------
# 4) Dataset @ 224x224, applies preprocessing.json (z-score if enabled)
# -------------------------------------------------------
class LCZ42Dataset(Dataset):
    def __init__(self, img_paths, labels, preprocess_json):
        self.img_paths = img_paths
        self.labels = labels
        with open(preprocess_json,'r') as f:
            self.meta = json.load(f)

        tfms = [
            transforms.Resize((224,224)),
            transforms.ToTensor(),  # 0–1
        ]
        if self.meta["zscore"]["enabled"]:
            mu = self.meta["zscore"]["mu"]
            sd = self.meta["zscore"]["sigma"]
            # If mu/sigma were computed for 3 channels, they must be length-3
            if len(mu) >= 3:
                mu = mu[:3]
                sd = sd[:3]
            tfms.append(transforms.Normalize(mean=mu, std=sd))
        self.transform = transforms.Compose(tfms)

    def __len__(self): return len(self.img_paths)
    def __getitem__(self, idx):
        from PIL import Image
        x = Image.open(self.img_paths[idx]).convert("RGB")
        y = self.labels[idx]
        return self.transform(x), y

# -------------------------------------------------------
# 5) Training (unchanged idea, tuned for 224)
# -------------------------------------------------------
def train_student(student, teacher, train_loader, val_loader, num_classes, epochs=30, lr=3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = student.to(device)
    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    criterion = DistillLoss(T=2.0, alpha=0.7)

    for epoch in range(epochs):
        student.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # teacher logits (N,C) from ONNX ensemble
            logits_teacher = teacher.predict_logits(x.detach().cpu().numpy())
            logits_teacher = torch.tensor(logits_teacher, device=device)
            # forward student
            logits_student = student(x)
            loss = criterion(logits_student, logits_teacher, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss {total_loss/len(train_loader):.4f}")

        # quick val acc
        student.eval(); correct=0; total=0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                pred = student(x).argmax(1)
                correct += (pred==y).sum().item()
                total += y.size(0)
        print(f" Val Acc {correct/total:.3f}")

    return student

# -------------------------------------------------------
# 6) Export ONNX (NCHW, dyn batch) for Metis
# -------------------------------------------------------
def export_student(student, out_path):
    student.eval()
    dummy = torch.randn(1,3,224,224)
    torch.onnx.export(student, dummy, out_path,
                      input_names=['input'], output_names=['logits'],
                      opset_version=17, dynamic_axes={'input':{0:'batch'}})
    print(f"[+] Exported student to {out_path}")

# -------------------------------------------------------
# Main usage example
# -------------------------------------------------------
if __name__ == "__main__":
    # teacher ONNX paths
    onnx_models = list(Path("deployment/onnx").glob("dense_*.onnx"))
    assert len(onnx_models) >= 1, "No teacher ONNX models found in deployment/onnx/"
    teacher = ONNXTeacher(onnx_models)

    # load your splits (fill these lists)
    train_paths, train_labels = [...], [...]
    val_paths,   val_labels   = [...], [...]
    preprocess_json = "deployment/onnx/preprocessing.json"

    train_ds = LCZ42Dataset(train_paths, train_labels, preprocess_json)
    val_ds   = LCZ42Dataset(val_paths,   val_labels,   preprocess_json)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    num_classes = len(json.load(open(preprocess_json))["labels"])

    # Build MobileNetV4-medium (timm) or fallback to EfficientNet-B0
    student = build_student(num_classes, arch="mobilenetv4m")
    student  = train_student(student, teacher, train_loader, val_loader, num_classes, epochs=30, lr=3e-4)

    export_student(student, "deployment/student_mbv4m_224.onnx")