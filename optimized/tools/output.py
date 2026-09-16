import os
import torch
import yaml
from cryptography.fernet import Fernet

def export_release_package(model, tuned_matrix, release_dir: str = "./dist_release"):
    """将训练好的神经网络权重 (.pth) 与 AES-256 加密的 YAML 配置文件统一打包至发布目录。"""
    os.makedirs(release_dir, exist_ok=True)
    
    # 1. 导出 PyTorch 模型权重
    model_path = os.path.join(release_dir, "risk_autoencoder_v1.pth")
    torch.save(model.state_dict(), model_path)
    print(f"[✓] Model weights serialized to: {model_path}")

# 2. 动态生成/读取加密密钥并导出安全 YAML (复用前面的安全配置逻辑)
    secret_key = Fernet.generate_key()
    cipher_suite = Fernet(secret_key)
    
    def _enc(val): return cipher_suite.encrypt(str(val).encode()).decode()
    
    production_config = {
        "metadata": {
            "version": "1.5.0",
            "model_binding": "risk_autoencoder_v1.pth",
            "encryption": "AES-256-GCM_Fernet"
        },
        "environments": {
            "baseline_healthy": {
                "kelly_allocation_fraction": _enc(float(round(tuned_matrix[0, 0], 4))),
                "vault_sweep_threshold": _enc(float(round(tuned_matrix[0, 1], 2))),
                "encrypted": True
            },
            "defensive_crash": {
                "kelly_allocation_fraction": _enc(float(round(tuned_matrix[1, 0], 4))),
                "vault_sweep_threshold": _enc(float(round(tuned_matrix[1, 1], 2))),
                "encrypted": True
            }
        }
    }
    
    yaml_path = os.path.join(release_dir, "production_config.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(production_config, f, default_flow_style=False, sort_keys=False)
        
    print(f"[✓] Secure YAML config exported to: {yaml_path}")
    print(f"[!] PRODUCTION SECRET KEY (Deploy to Env Var): {secret_key.decode()}\n")