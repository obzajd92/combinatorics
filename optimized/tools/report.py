import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_historical_convergence_v2(
    healthy_path: str, crash_path: str, output_dir: str = "./plots"
) -> None:
    """绘制参数历史收敛轨迹，自动高亮并标注绝对最优迭代点，最终导出为高分辨率 PNG。"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载并清洗数据
    df_h = pd.read_csv(healthy_path)
    df_c = pd.read_csv(crash_path)
    viable_h = df_h[df_h["Status"] != "REJECTED_HARD"].reset_index(drop=True)
    viable_c = df_c[df_c["Status"] != "REJECTED_HARD"].reset_index(drop=True)

    # 2. 创建画布
    fig, (ax_h, ax_c) = plt.subplots(2, 1, figsize=(14, 12))

    def draw_scenario_and_highlight(ax, df, title_name):
        if df.empty:
            return

        # 找到绝对最优点的索引位置
        best_idx = df["Adjusted_Loss"].idxmin()
        best_loss = df.loc[best_idx, "Adjusted_Loss"]
        best_kelly = df.loc[best_idx, "Kelly_Fraction"]
        best_vault = df.loc[best_idx, "Vault_Threshold"]

        # --- 绘制主轴：Adjusted Loss ---
        (line_loss,) = ax.plot(
            df.index,
            df["Adjusted_Loss"],
            color="#d62728",
            linewidth=2,
            label="Adjusted Loss",
        )
        ax.set_ylabel("Adjusted Loss", color="#d62728")
        ax.tick_params(axis="y", labelcolor="#d62728")
        ax.grid(True, linestyle=":", alpha=0.6)

        # 🌟 高亮最优 Loss 点
        ax.scatter(
            best_idx,
            best_loss,
            color="#ffd700",
            edgecolors="#d62728",
            s=250,
            marker="*",
            zorder=5,
            label="Optimal Configuration",
        )
        # 添加最优点的 Loss 数值标签
        ax.annotate(
            f"Best Loss: {best_loss:.4f}\n(Iter {best_idx})",
            xy=(best_idx, best_loss),
            xytext=(15, 15),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1),
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff2cc", ec="#d62728", alpha=0.9),
            fontweight="bold",
        )

        # --- 绘制孪生轴 1：Kelly Fraction ---
        ax_k = ax.twinx()
        (line_k,) = ax_k.plot(
            df.index,
            df["Kelly_Fraction"],
            color="#1f77b4",
            linestyle="--",
            alpha=0.7,
            label="Kelly Fraction",
        )
        ax_k.set_ylabel("Kelly Fraction", color="#1f77b4")
        ax_k.tick_params(axis="y", labelcolor="#1f77b4")

        # 🌟 高亮最优 Kelly 点并标值
        ax_k.scatter(best_idx, best_kelly, color="#1f77b4", s=60, marker="o", zorder=5)
        ax_k.text(
            best_idx,
            best_kelly,
            f" K:{best_kelly:.4f} ",
            color="#1f77b4",
            ha="right",
            va="bottom",
            fontweight="bold",
        )

        # --- 绘制孪生轴 2：Vault Threshold ---
        ax_v = ax.twinx()
        ax_v.spines["right"].set_position(("outward", 65))
        (line_v,) = ax_v.plot(
            df.index,
            df["Vault_Threshold"],
            color="#2ca02c",
            linestyle="-.",
            alpha=0.7,
            label="Vault Threshold",
        )
        ax_v.set_ylabel("Vault Sweep Threshold ($)", color="#2ca02c")
        ax_v.tick_params(axis="y", labelcolor="#2ca02c")

        # 🌟 高亮最优 Vault 点并标值
        ax_v.scatter(best_idx, best_vault, color="#2ca02c", s=60, marker="s", zorder=5)
        ax_v.text(
            best_idx,
            best_vault,
            f" V:${best_vault:,.2f} ",
            color="#2ca02c",
            ha="left",
            va="top",
            fontweight="bold",
        )

        # 标题与合并图例
        ax.set_title(f"Optimization Trajectory: {title_name}", fontsize=14, pad=15)
        ax.set_xlabel("Viable Iteration Index")

        # 合并三个轴的图例到一个框内
        lines = [line_loss, line_k, line_v]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper right")

    # 执行绘制
    draw_scenario_and_highlight(ax_h, viable_h, "Baseline Scenario")
    draw_scenario_and_highlight(ax_c, viable_c, "Defensive Crash Scenario")

    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(output_dir, "optimized_parameter_convergence.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[✓] Enhanced trajectory plot with highlighters saved to: {save_path}")
    plt.close()
