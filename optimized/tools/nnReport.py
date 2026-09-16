import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 保证实验可重复性
torch.manual_seed(42)
np.random.seed(42)


# ==========================================
# 🧠 1. 自编码器神经网络定义（带隐空间截取）
# ==========================================
class RiskAutoencoderNN(nn.Module):

    def __init__(self, input_dim: int = 2, latent_dim: int = 1):
        super(RiskAutoencoderNN, self).__init__()

        # 编码器 (Encoder)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8), nn.Tanh(), nn.Linear(8, latent_dim)
        )

        # 解码器 (Decoder)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8), nn.Tanh(), nn.Linear(8, input_dim), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent  # 同时返回重构结果和隐空间变量


# ==========================================
# 🔄 2. 模拟训练与数据记录管线
# ==========================================
def train_and_report_nn(epochs: int = 200):
    # 模拟历史大量的历史风控参数作为训练集 (500个样本, 包含 Kelly 和 归一化后的 Vault)
    # 假设真实的流形规律是：Vault 越大，Kelly 越保守
    mock_vault = np.random.uniform(5000, 100000, (500, 1))
    # 物理归一化 (Vault 除以 100,000 限制在 0~1)
    mock_vault_scaled = mock_vault / 100000.0
    mock_kelly = 0.4 / (1.0 + 2.0 * mock_vault_scaled) + np.random.normal(
        0, 0.02, (500, 1)
    )
    mock_kelly = np.clip(mock_kelly, 0.01, 0.5)

    train_data = np.hstack((mock_kelly, mock_vault_scaled))
    X_train = torch.tensor(train_data, dtype=torch.float32)

    # 实例化模型、损失函数与优化器
    model = RiskAutoencoderNN(input_dim=2, latent_dim=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 用于绘图的数据记录器
    epoch_history = []
    loss_history = []
    kelly_history = []  # 记录某一个典型测试样本的 Kelly 演变过程
    vault_history = []  # 记录该样本的 Vault 演变过程
    latent_history = []  # 记录该样本在隐空间的映射值

    # 选择索引为 0 的样本作为跟踪观察的目标对象
    test_sample = X_train[0:1]
    true_kelly = test_sample[0, 0].item()
    true_vault = test_sample[0, 1].item() * 100000.0

    print("=======================================================")
    print(" 🏋️ STARTING AUTOENCODER NEURAL NETWORK TRAINING")
    print("=======================================================")

    # 开始训练循环
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # 前向传播
        predictions, _ = model(X_train)
        loss = criterion(predictions, X_train)

        # 反向传播与优化
        loss.backward()
        optimizer.step()

        # 捕获跟踪样本的状态
        model.eval()
        with torch.no_grad():
            pred_sample, latent_sample = model(test_sample)
            # 逆归一化还原真实物理尺度
            pred_k = pred_sample[0, 0].item()
            pred_v = pred_sample[0, 1].item() * 100000.0
            latent_val = latent_sample[0, 0].item()

        # 记录历史
        epoch_history.append(epoch)
        loss_history.append(loss.item())
        kelly_history.append(pred_k)
        vault_history.append(pred_v)
        latent_history.append(latent_val)

        if epoch % 40 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{epochs} | Loss: {loss.item():.6f} | Latent Hinge: {latent_val:+.4f}"
            )

    print("=======================================================\n")

    # ==========================================
    # 📊 3. Matplotlib 神经网络报告生成
    # ==========================================
    # 创建 1行2列 的精美大图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- 图表A: Epochs vs Variables (真实物理参数收敛) ---
    color_k = "#1f77b4"
    ax1.plot(
        epoch_history,
        kelly_history,
        color=color_k,
        lw=2.5,
        label="Tuned Kelly Fraction",
    )
    ax1.axhline(
        true_kelly,
        color=color_k,
        linestyle=":",
        alpha=0.7,
        label="Target Raw Kelly",
    )
    ax1.set_xlabel("Epochs", fontsize=11)
    ax1.set_ylabel("Kelly Allocation Fraction", color=color_k, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_k)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 共享X轴，创建右侧 Y 轴展示 Vault
    ax1_twin = ax1.twinx()
    color_v = "#2ca02c"
    ax1_twin.plot(
        epoch_history,
        vault_history,
        color=color_v,
        lw=2.5,
        linestyle="-.",
        label="Tuned Vault Threshold",
    )
    ax1_twin.axhline(
        true_vault,
        color=color_v,
        linestyle=":",
        alpha=0.7,
        label="Target Raw Vault",
    )
    ax1_twin.set_ylabel("Vault Sweep Threshold ($)", color=color_v, fontsize=11)
    ax1_twin.tick_params(axis="y", labelcolor=color_v)

    # 合并双 Y 轴图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    ax1.set_title("Epochs vs Physical Variables (Convergence Tracking)", fontsize=13)

    # --- 图表B: Epochs vs Latent Variable (隐空间压缩轨迹) ---
    color_l = "#9467bd"
    ax2.plot(
        epoch_history,
        latent_history,
        color=color_l,
        lw=3,
        label="Latent Feature Vector (z)",
    )
    ax2.set_xlabel("Epochs", fontsize=11)
    ax2.set_ylabel("Latent Space Coordinate Value", color=color_l, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_l)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 额外在右侧副轴画出 Loss 下降，以对比隐空间锁定与整体收敛的关系
    ax2_twin = ax2.twinx()
    ax2_twin.plot(
        epoch_history,
        loss_history,
        color="#d62728",
        alpha=0.3,
        label="Training MSE Loss",
    )
    ax2_twin.set_ylabel("Total Training MSE Loss", color="#d62728")
    ax2_twin.tick_params(axis="y", labelcolor="#d62728")

    lines_3, labels_3 = ax2.get_legend_handles_labels()
    lines_4, labels_4 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_3 + lines_4, labels_3 + labels_4, loc="upper right")
    ax2.set_title("Epochs vs Latent Variable (Manifold Compression)", fontsize=13)

    plt.tight_layout()

    # 自动保存到本地物理图片
    save_plot_path = "./nn_reporting_epochs_vs_latent.png"
    plt.savefig(save_plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(
        f"[✓] Matplotlib NN Pipeline Report successfully exported to: {save_plot_path}"
    )


# 触发执行
if __name__ == "__main__":
    train_and_report_nn(epochs=250)
