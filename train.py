import os
import glob
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from trackformer import MobileViT_XT

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class NightwatchDataset(Dataset):
    def __init__(self, data_dir="RAW_DATA", num_frames=16, img_w=640, img_h=480, target_size=64):
        self.data_dir = data_dir
        self.clips = glob.glob(os.path.join(data_dir, "clip_*"))
        self.num_frames = num_frames
        self.img_w = img_w
        self.img_h = img_h
        self.target_size = target_size
        
        print(f"[*] NightwatchDataset: Se encontraron {len(self.clips)} clips para entrenamiento.")

    def __len__(self):
        return len(self.clips)

    def _calc_local_std(self, x, kernel_size=3):
        # x shape: (1, 1, H, W)
        mean_x = F.avg_pool2d(x, kernel_size, stride=1, padding=1)
        mean_x2 = F.avg_pool2d(x**2, kernel_size, stride=1, padding=1)
        variance = torch.clamp(mean_x2 - mean_x**2, min=0.0)
        return torch.sqrt(variance)

    def __getitem__(self, idx):
        clip_path = self.clips[idx]
        
        frames = []
        gts = []
        
        # Cargar los 16 frames
        for t in range(self.num_frames):
            bin_file = os.path.join(clip_path, f"frame_{t}.bin")
            gt_file = os.path.join(clip_path, f"gt_{t}.txt")
            
            # Leer matriz Husimi-Q cruda (640x480, float32)
            with open(bin_file, "rb") as f:
                raw_data = np.fromfile(f, dtype=np.float32)
            
            # Formatear a tensor (1, 1, 480, 640)
            tensor_2d = torch.from_numpy(raw_data).view(1, 1, self.img_h, self.img_w)
            
            # Downsample a 64x64 para la red
            tensor_64 = F.adaptive_avg_pool2d(tensor_2d, (self.target_size, self.target_size))
            frames.append(tensor_64)
            
            # Leer GT del ÚLTIMO frame temporal (la red predice el estado final)
            if t == self.num_frames - 1:
                with open(gt_file, "r") as f:
                    vals = f.read().strip().split(",")
                    # vals: [x, y, vx, vy, confidence]
                    # Normalizar coordenadas de 640x480 al espacio 64x64
                    x_norm = float(vals[0]) * (self.target_size / self.img_w)
                    y_norm = float(vals[1]) * (self.target_size / self.img_h)
                    vx_norm = float(vals[2]) * (self.target_size / self.img_w)
                    vy_norm = float(vals[3]) * (self.target_size / self.img_h)
                    confidence = float(vals[4])
                    
                    gt_tensor = torch.tensor([x_norm, y_norm, vx_norm, vy_norm, confidence], dtype=torch.float32)
        
        # Ensamblar canales
        t_frames = torch.cat(frames, dim=1) # (1, 16, 64, 64)
        
        # Canal 1: Densidad (C1)
        c1 = t_frames
        
        # Canal 2: Diferencial temporal (C2 = T - T-1)
        c2 = torch.zeros_like(t_frames)
        c2[:, 1:, :, :] = t_frames[:, 1:, :, :] - t_frames[:, :-1, :, :]
        
        # Canal 3: Desviación Estándar local (C3)
        # Procesamos frame a frame
        c3_list = []
        for t in range(self.num_frames):
            frame_t = t_frames[:, t:t+1, :, :]
            c3_list.append(self._calc_local_std(frame_t))
        c3 = torch.cat(c3_list, dim=1) # (1, 16, 64, 64)
        
        # Tensor final (3, 16, 64, 64)
        input_cube = torch.cat([c1, c2, c3], dim=0) # (3, 16, 64, 64)
        
        return input_cube, gt_tensor

class NightwatchLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss(reduction='none') # No reducimos para poder aplicar la máscara
        
    def forward(self, pred_p_det, pred_coords, target):
        # Target: [B, 5] -> (x, y, vx, vy, active)
        target_coords = target[:, :4]
        target_active = target[:, 4].unsqueeze(1) # (B, 1)
        
        # 1. Loss de Detección (Probabilidad de objeto)
        loss_det = self.bce(pred_p_det, target_active)
        
        # 2. Loss de Regresión (Solo si target_active == 1)
        loss_reg_raw = self.mse(pred_coords, target_coords) # (B, 4)
        loss_reg_masked = loss_reg_raw * target_active      # Multiplicamos por la máscara binaria
        loss_reg = loss_reg_masked.sum() / (target_active.sum() + 1e-6) # Promedio sobre las muestras activas
        
        # Combinación (Damos más peso a las coordenadas)
        total_loss = loss_det + 5.0 * loss_reg
        return total_loss, loss_det, loss_reg

def train_model(epochs=10, batch_size=2):
    dataset = NightwatchDataset("RAW_DATA")
    if len(dataset) == 0:
        print("[!] No hay datos para entrenar. Ejecuta dataset_generator.py primero.")
        return
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Entrenando en hardware: {device}")
    
    model = MobileViT_XT().to(device)
    criterion = NightwatchLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    print("\n--- INICIANDO LOOP DE ENTRENAMIENTO ---")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        det_l = 0.0
        reg_l = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            p_det, coords = model(inputs)
            loss, l_det, l_reg = criterion(p_det, coords, targets)
            
            loss.backward()
            
            # Gradient clipping para estabilidad del RNN (BiLSTM)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            det_l += l_det.item()
            reg_l += l_reg.item()
            
        print(f"Época [{epoch+1}/{epochs}] | Loss Total: {epoch_loss/len(dataloader):.4f} | Det: {det_l/len(dataloader):.4f} | Reg: {reg_l/len(dataloader):.4f}")
        
    print("--- ENTRENAMIENTO COMPLETADO ---")
    
    # Guardar los pesos
    torch.save(model.state_dict(), "nightwatch_mobilevit.pth")
    print("[OK] Pesos guardados en 'nightwatch_mobilevit.pth'")

if __name__ == "__main__":
    # Configuramos épocas bajas para prueba de humo (smoke test)
    train_model(epochs=5, batch_size=2)
