def plot_drift_slope(k_healthy, k_crash, v_healthy, v_crash):
    """绘制前后对比折线图（斜率图）展示参数漂移趋势"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    scenarios = ["Baseline", "Defensive Crash"]

    # 1. Kelly Fraction 趋势线
    ax1.plot(
        scenarios, [k_healthy, k_crash], marker="o", linewidth=2, color="#1f77b4"
    )
    ax1.set_title("Kelly Fraction Drift")
    ax1.set_ylabel("Allocation Fraction")
    ax1.grid(True, linestyle="--", alpha=0.6)
    # 为点添加数据标签
    ax1.text(0, k_healthy, f" {k_healthy:.4f}", ha="right", va="center")
    ax1.text(1, k_crash, f" {k_crash:.4f}", ha="left", va="center")

    # 2. Vault Threshold 趋势线
    ax2.plot(
        scenarios, [v_healthy, v_crash], marker="s", linewidth=2, color="#ff7f0e"
    )
    ax2.set_title("Vault Sweep Threshold Drift")
    ax2.set_ylabel("Threshold ($)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    # 为点添加数据标签
    ax2.text(0, v_healthy, f" ${v_healthy:,.2f} ", ha="right", va="center")
    ax2.text(1, v_crash, f" ${v_crash:,.2f} ", ha="left", va="center")

    plt.tight_layout()
    plt.show()
def plot_drift_bars(k_healthy, k_crash, v_healthy, v_crash):
    """绘制对比条形图展示参数绝对值差异"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(2)
    labels = ["Baseline", "Defensive Crash"]
    width = 0.4

    # 1. Kelly Fraction 条形图
    bars1 = ax1.bar(
        labels, [k_healthy, k_crash], width=width, color=["#2ca02c", "#d62728"]
    )
    ax1.set_title("Kelly Fraction Comparison")
    ax1.set_ylabel("Allocation Fraction")
    ax1.bar_label(bars1, fmt="%.4f", padding=3)

    # 2. Vault Threshold 条形图
    bars2 = ax2.bar(
        labels, [v_healthy, v_crash], width=width, color=["#2ca02c", "#d62728"]
    )
    ax2.set_title("Vault Sweep Threshold Comparison")
    ax2.set_ylabel("Threshold ($)")
    ax2.bar_label(bars2, labels=[f"${v:,.2f}" for v in [v_healthy, v_crash]], padding=3)

    plt.tight_layout()
    plt.show()
