import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# KĦAOS-TRACKFORMER: MobileViT-XT (eXtended Temporal) Architecture
# Basado en el blueprint de la División de Arquitectura de IA de NIGHTWATCH
# -----------------------------------------------------------------------------

class SqueezeExcitation(nn.Module):
    """SE-attention para suprimir ruido centelleante atmosférico"""
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Conv3d(in_channels, in_channels // reduction, 1)
        self.fc2 = nn.Conv3d(in_channels // reduction, in_channels, 1)
        
    def forward(self, x):
        # Global Average Pooling espacial y temporal
        scale = x.mean(dim=(2, 3, 4), keepdim=True)
        scale = F.relu6(self.fc1(scale))  # Usamos ReLU6 para cuantización INT8
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale

class InvertedResidual3D(nn.Module):
    """Bloque MobileNetV3-IR adaptado a 3D (Espaciotemporal)"""
    def __init__(self, in_c, out_c, expansion, stride):
        super().__init__()
        hidden_c = in_c * expansion
        self.use_res_connect = stride == 1 and in_c == out_c

        layers = []
        if expansion != 1:
            layers.extend([
                nn.Conv3d(in_c, hidden_c, kernel_size=1, bias=False),
                nn.BatchNorm3d(hidden_c),
                nn.ReLU6(inplace=True)
            ])
            
        layers.extend([
            # Depthwise 3D Conv
            nn.Conv3d(hidden_c, hidden_c, kernel_size=3, stride=stride, padding=1, groups=hidden_c, bias=False),
            nn.BatchNorm3d(hidden_c),
            nn.ReLU6(inplace=True),
            SqueezeExcitation(hidden_c),
            # Pointwise linear
            nn.Conv3d(hidden_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_c)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

class FSA_Block(nn.Module):
    """Factorized Spatiotemporal Attention (FSA)"""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        
        # FSA-1: Spatial Self-Attention (Intra-Frame)
        self.spatial_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.spatial_proj = nn.Linear(dim, dim)
        
        # FSA-2: Temporal Differential Attention (Inter-Frame)
        self.temporal_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.temporal_proj = nn.Linear(dim, dim)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        B, C, T, H, W = x.shape
        # Preparar para atención espacial: (B*T, H*W, C)
        xs = x.permute(0, 2, 3, 4, 1).contiguous().view(B*T, H*W, C)
        
        # --- FSA-1: Atención Espacial ---
        qkv_s = self.spatial_qkv(self.norm1(xs)).chunk(3, dim=-1)
        q_s, k_s, v_s = [t.view(B*T, H*W, self.num_heads, C // self.num_heads).transpose(1, 2) for t in qkv_s]
        
        # Softmax con Temperatura escalada para INT8
        attn_s = (q_s @ k_s.transpose(-2, -1)) / ((C // self.num_heads) ** 0.5)
        attn_s = F.softmax(attn_s / 2.0, dim=-1) # / 2.0 Temp scaling
        
        xs_out = (attn_s @ v_s).transpose(1, 2).reshape(B*T, H*W, C)
        xs = xs + self.spatial_proj(xs_out)
        
        # Volver al tensor 3D original
        x_spa = xs.view(B, T, H, W, C).permute(0, 4, 1, 2, 3)
        
        # --- FSA-2: Atención Temporal Diferencial ---
        # Pooling espacial para obtener un token por frame: (B, T, C)
        xt = x_spa.mean(dim=(3, 4)).permute(0, 2, 1) # (B, T, C)
        
        # Crear Embeddings Diferenciales (e_t - e_{t-1})
        xt_diff = torch.zeros_like(xt)
        xt_diff[:, 1:, :] = xt[:, 1:, :] - xt[:, :-1, :]
        
        qkv_t = self.temporal_qkv(self.norm2(xt_diff)).chunk(3, dim=-1)
        q_t, k_t, v_t = [t.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) for t in qkv_t]
        
        attn_t = (q_t @ k_t.transpose(-2, -1)) / ((C // self.num_heads) ** 0.5)
        
        # Máscara Causal para tiempo real
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        attn_t = attn_t.masked_fill(mask == 0, float('-inf'))
        attn_t = F.softmax(attn_t / 2.0, dim=-1)
        
        xt_out = (attn_t @ v_t).transpose(1, 2).reshape(B, T, C)
        xt_out = self.temporal_proj(xt_out)
        
        # Expandir la atención temporal de vuelta a la rejilla espacial
        # xt_out es (B, T, C) -> transpose(1, 2) -> (B, C, T, 1, 1)
        x_out = x_spa + xt_out.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        return x_out

class MobileViT_XT(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 3 x 16 x 64 x 64 (Canales diferenciales, Tiempo, H, W)
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU6(inplace=True)
        ) # Sale: 16 x 16 x 32 x 32
        
        # Stage 1: Spatial CNN
        self.stage1 = nn.Sequential(
            InvertedResidual3D(16, 24, expansion=4, stride=(1, 1, 1)),
            InvertedResidual3D(24, 24, expansion=4, stride=(1, 1, 1))
        )
        
        # Stage 2: Spatial CNN
        self.stage2 = nn.Sequential(
            InvertedResidual3D(24, 48, expansion=4, stride=(1, 2, 2)), # Baja res a 16x16
            InvertedResidual3D(48, 48, expansion=4, stride=(1, 1, 1)),
            InvertedResidual3D(48, 48, expansion=4, stride=(1, 1, 1))
        ) # Sale: 48 x 16 x 16 x 16
        
        # Stage 3: Temporal Transformer (FSA)
        # Bajamos la dimensionalidad para meterlo al Transformer
        self.proj_to_transformer = nn.Conv3d(48, 128, kernel_size=1)
        self.fsa1 = FSA_Block(128)
        self.fsa2 = FSA_Block(128)
        self.proj_from_transformer = nn.Conv3d(128, 48, kernel_size=1)
        
        # Stage 4: Temporal CNN (Pooling Temporal)
        self.stage4 = nn.Sequential(
            InvertedResidual3D(48, 64, expansion=4, stride=(2, 2, 2)), # Baja tiempo a 8, res a 8x8
            InvertedResidual3D(64, 64, expansion=4, stride=(1, 1, 1))
        ) # Sale: 64 x 8 x 8 x 8
        
        # --- DETECTION HEAD (Confianza) ---
        self.det_head_conv = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU6(inplace=True)
        )
        self.det_head_temp = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.det_head_out = nn.Linear(32, 1)
        
        # --- REGRESSION HEAD (Coordenadas y Velocidad) ---
        self.reg_head_conv = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU6(inplace=True)
        )
        self.reg_blstm = nn.LSTM(input_size=32, hidden_size=16, num_layers=1, batch_first=True, bidirectional=True)
        self.reg_out = nn.Linear(32, 4) # x, y, vx, vy

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        
        x = self.proj_to_transformer(x)
        x = self.fsa1(x)
        x = self.fsa2(x)
        x = self.proj_from_transformer(x)
        
        x = self.stage4(x)
        
        # --- Heads Processing ---
        # Detection Head
        d = self.det_head_conv(x)
        d = d.mean(dim=(3, 4)) # Global Spatial Pool -> (B, 32, T)
        d = self.det_head_temp(d)
        d = d.mean(dim=-1) # Average over time -> (B, 32)
        p_det = torch.sigmoid(self.det_head_out(d)) # (B, 1)
        
        # Regression Head
        r = self.reg_head_conv(x)
        r = r.mean(dim=(3, 4)) # (B, 32, T)
        r = r.transpose(1, 2) # (B, T, 32)
        r_seq, _ = self.reg_blstm(r) # BiLSTM over time
        r_last = r_seq[:, -1, :] # Tomamos el último estado temporal -> (B, 32)
        coords = self.reg_out(r_last) # (B, 4) -> x, y, vx, vy
        
        # Confidence-gated output: Seed vector
        return p_det, coords * p_det

if __name__ == "__main__":
    print("[NIGHTWATCH-TRACKFORMER] Inicializando arquitectura MobileViT-XT...")
    model = MobileViT_XT()
    
    # Tensor de prueba: Lote 1, 3 canales, 16 frames, 64x64 pixeles
    dummy_input = torch.randn(1, 3, 16, 64, 64)
    
    p_det, coords = model(dummy_input)
    
    print("OK: Inferencia hacia adelante (Forward Pass) ejecutada con exito.")
    print(f"Salida de Deteccion (p_det): {p_det.item():.4f}")
    print(f"Salida de Regresion [x, y, vx, vy]: {coords.detach().numpy()[0]}")
    print("\nEl modelo esta listo para exportacion a TensorRT INT8 (nvinfer1).")
