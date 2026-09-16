import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_historical_convergence(
    healthy_path: str, crash_path: str, output_dir: str = "./plots"
) -> None:
    """读取所有可行迭代数据，绘制参数随优化进程的历史收敛轨迹，并保存为高分辨率 PNG。"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载并清洗数据（过滤掉硬拒绝的迭代）
    df_h = pd.read_csv(healthy_path)
    df_c = pd.read_csv(crash_path)
    viable_h = df_h[df_h["Status"] != "REJECTED_HARD"].reset_index(drop=True)
    viable_c = df_c[df_c["Status"] != "REJECTED_HARD"].reset_index(drop=True)

    # 2. 创建画布 (2行1列：上方展示 Baseline 演变，下方展示 Crash 演变)
    fig, (ax_h, ax_c) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

    def draw_scenario_trajectory(ax, df, title_name):
        # 基础轴 (X轴为可行迭代的序号)：绘制 Adjusted Loss
        ax.plot(
            df.index,
            df["Adjusted_Loss"],
            color="#d62728",
            linewidth=2,
            label="Adjusted Loss (Left)",
        )
        ax.set_ylabel("Adjusted Loss", color="#d62728")
        ax.tick_params(axis="y", labelcolor="#d62728")
        ax.grid(True, linestyle=":", alpha=0.6)

        # 孪生轴 1 (右轴)：绘制 Kelly Fraction
        ax_k = ax.twinx()
        ax_k.plot(
            df.index,
            df["Kelly_Fraction"],
            color="#1f77b4",
            linestyle="--",
            alpha=0.8,
            label="Kelly Fraction (Right)",
        )
        ax_k.set_ylabel("Kelly Fraction", color="#1f77b4")
        ax_k.tick_params(axis="y", labelcolor="#1f77b4")

        # 孪生轴 2 (侧边偏移轴)：绘制 Vault Threshold
        ax_v = ax.twinx()
        # 将第三个 Y 轴向右移动
        ax_v.spines["right"].set_position(("outward", 60))
        ax_v.plot(
            df.index,
            df["Vault_Threshold"],
            color="#2ca02c",
            linestyle="-.",
            alpha=0.8,
            label="Vault Threshold (Far Right)",
        )
        ax_v.set_ylabel("Vault Sweep Threshold ($)", color="#2ca02c")
        ax_v.tick_params(axis="y", labelcolor="#2ca02c")

        # 标题与标签
        ax.set_title(f"Optimization Trajectory & Convergence: {title_name}")
        ax.set_xlabel("Viable Iteration Count")

    # 绘制两个场景
    draw_scenario_trajectory(ax_h, viable_h, "Baseline Scenario")
    draw_scenario_trajectory(ax_c, viable_c, "Defensive Crash Scenario")

    plt.tight_layout()

    # 3. 自动导出并保存图片
    save_path = os.path.join(output_dir, "parameter_convergence_trajectory.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[✓] Convergence trajectory plot successfully saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    healthy_log = "path/to/healthy_telemetry.csv"
    crash_log = "path/to/crash_telemetry.csv"

    # 1. 控制台打印量化指标
    analyze_parameter_drift(healthy_log, crash_log)

    # 2. 绘制多轮迭代的整体收敛轨迹并自动保存本地
    plot_historical_convergence(healthy_log, crash_log, output_dir="./output_plots")

