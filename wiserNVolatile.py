import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Set seed for reproducibility of volatile fluctuations
np.random.seed(42)

# 1. Define the input set of intervals (Hyperparameter)
input_set = [(1.0, 2.5), (3.5, 5.0)] 

# 2. Define Bounds (Infimum and Supremum of the set)
inf_E = min(start for start, end in input_set)
sup_E = max(end for start, end in input_set)

# 3. Indicator (Characteristic) function of the set E
def indicator_E(x):
    return 1.0 if any(start <= x <= end for start, end in input_set) else 0.0

indicator_E_vec = np.vectorize(indicator_E)

# 4. Define the baseline function & Volatile function
def f_base(x):
    return x * np.sin(x)

def f_volatile(x, noise_level=1.8):
    """Adds high-frequency oscillatory volatility to the base curve."""
    # Simulates a highly volatile asset path or noise function
    return f_base(x) + noise_level * np.sin(20 * x) * np.cos(5 * x)

def integrand_volatile(x):
    return f_volatile(x) * indicator_E(x)

integrand_volatile_vec = np.vectorize(integrand_volatile)

# 5. Track cumulative integration iterations with high-resolution steps
x_steps = np.linspace(inf_E, sup_E, 1000)
dx = x_steps[1] - x_steps[0]
running_integral = 0.0
iterations_x = []
iterations_y = []

for x in x_steps:
    running_integral += integrand_volatile(x) * dx
    iterations_x.append(x)
    iterations_y.append(running_integral)

# Compute exact integral using quadrature pieces across the disjoint sets
precise_integral = 0.0
for start, end in input_set:
    val, _ = quad(f_volatile, start, end)
    precise_integral += val

print(f"--- Volatile Lebesgue Measure Domain ---")
print(f"Infimum (inf E): {inf_E}")
print(f"Supremum (sup E): {sup_E}")
print(f"Total Volatile Integral Value: {precise_integral:.4f}\n")


# ==========================================
# PLOTTING THE VOLATILE SCENARIO
# ==========================================
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
x_domain = np.linspace(inf_E - 1, sup_E + 1, 1000)

# Plot 1: The Volatile Function & Shaded Integration Area
axs.plot(x_domain, f_base(x_domain), label="Base Curve (No Noise)", color="gray", linestyle="--", alpha=0.7)
axs.plot(x_domain, integrand_volatile_vec(x_domain), label="Volatile Integrand $f_{vol}(x)\chi_E(x)$", color="blue", lw=1.5)

# Shade integration area in light blue with red integration bounds
for start, end in input_set:
    axs.axvline(x=start, color="red", linestyle="--", lw=1.5)
    axs.axvline(x=end, color="red", linestyle="--", lw=1.5)
    x_fill = np.linspace(start, end, 500)
    axs.fill_between(x_fill, f_volatile(x_fill), color="lightblue", alpha=0.6)
# Plot 2: Volatile Iterations (Cumulative Convergence Graph)
axs.plot(iterations_x, iterations_y, color="purple", label="Volatile Cumulative Sum")
axs.axhline(y=precise_integral, color="green", linestyle="-.", label=f"Final Value ({precise_integral:.2f})")
axs.set_title("Integration Iterations (Notice the jagged step paths)")
axs.legend()
axs.grid(True)

# Plot 3: Analogous PDF (Normalized Indicator Function of Set E)
measure_E = sum(end - start for start, end in input_set)
pdf_y = indicator_E_vec(x_domain) / measure_E

axs.plot(x_domain, pdf_y, color="darkorange", lw=2, label="Domain PDF")
axs.fill_between(x_domain, pdf_y, color="orange", alpha=0.2)
axs.set_title("Lebesgue Measure Set PDF")
axs.legend()
axs.grid(True)

# Plot 4: Analogous CDF (Cumulative Measure / Total Measure)
cdf_y = []
for x in x_domain:
    cum_measure = 0.0
    for start, end in input_set:
        if x > start:
            cum_measure += min(x, end) - start
    cdf_y.append(cum_measure / measure_E)
axs.plot(x_domain, cdf_y, color="green", lw=2, label="Domain CDF")
axs.set_title("Lebesgue Measure Set CDF")
axs.legend()
axs.grid(True)

plt.tight_layout()
plt.show()

