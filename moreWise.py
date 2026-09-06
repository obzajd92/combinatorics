import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# 1. Define the input set of intervals (Hyperparameter)
# This represents a multiple set of points forming a measurable set E
input_set = [(1.0, 2.5), (3.5, 5.0)] 

# 2. Define Bounds (Infimum and Supremum of the set)
inf_E = min(start for start, end in input_set)
sup_E = max(end for start, end in input_set)

# 3. Indicator (Characteristic) function of the set E
def indicator_E(x):
    """Returns 1 if x is inside the measurable set E, else 0."""
    return 1.0 if any(start <= x <= end for start, end in input_set) else 0.0

# Vectorized version for plotting
indicator_E_vec = np.vectorize(indicator_E)

# 4. Define the function to integrate
def f(x):
    """Example formula to integrate: f(x) = x * sin(x)"""
    return x * np.sin(x)

def integrand(x):
    """The actual function integrated under the Lebesgue measure approach."""
    return f(x) * indicator_E(x)

integrand_vec = np.vectorize(integrand)

# 5. Solve the Integral & track step-by-step iterations
total_integral = 0.0
iterations_x = []
iterations_y = []

# Numerical verification / tracking of iterations across the domain
x_steps = np.linspace(inf_E, sup_E, 100)
running_integral = 0.0
dx = x_steps[1] - x_steps[0]

for x in x_steps:
    running_integral += integrand(x) * dx
    iterations_x.append(x)
    iterations_y.append(running_integral)

# Compute precise integral using scipy.quad over the disjoint pieces
precise_integral = 0.0
for start, end in input_set:
    val, _ = quad(f, start, end)
    precise_integral += val

print(f"--- Lebesgue Measure Domain Bounds ---")
print(f"Infimum (inf E): {inf_E}")
print(f"Supremum (sup E): {sup_E}")
print(f"Total Precise Integral Value: {precise_integral:.4f}\n")


# ==========================================
# PLOTTING
# ==========================================
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
x_domain = np.linspace(inf_E - 1, sup_E + 1, 1000)

# Plot 1: The Function & Shaded Integration Area
axs[0, 0].plot(x_domain, f(x_domain), label=r"$f(x) = x \sin(x)$", color="black", alpha=0.5)
axs[0, 0].plot(x_domain, integrand_vec(x_domain), label=r"Integrand $f(x)\chi_E(x)$", color="blue", lw=2)

# Shade integration area in light blue with red integration bounds
for start, end in input_set:
    # Red lines for bounds
    axs[0, 0].axvline(x=start, color="red", linestyle="--", lw=1.5)
    axs[0, 0].axvline(x=end, color="red", linestyle="--", lw=1.5)
    # Light blue fill
    x_fill = np.linspace(start, end, 200)
    axs[0, 0].fill_between(x_fill, f(x_fill), color="lightblue", alpha=0.6)

axs[0, 0].set_title("Integration Area (Light Blue) & Red Bounds")
axs[0, 0].legend()
axs[0, 0].grid(True)

# Plot 2: Iterations / Numerical Convergence Plot
axs[0, 1].plot(iterations_x, iterations_y, color="purple", marker="o", markersize=3, label="Cumulative Sum")
axs[0, 1].axhline(y=precise_integral, color="green", linestyle="-.", label=f"Final Value ({precise_integral:.2f})")
axs[0, 1].set_title("Integration Iterations (Cumulative Convergence)")
axs[0, 1].legend()
axs[0, 1].grid(True)

# Plot 3: Analogous PDF (Normalized Indicator Function of Set E)
# Total lebesgue measure (length) of the set E
measure_E = sum(end - start for start, end in input_set)
pdf_y = indicator_E_vec(x_domain) / measure_E

axs[1, 0].plot(x_domain, pdf_y, color="darkorange", lw=2, label="Normalized PDF")
axs[1, 0].fill_between(x_domain, pdf_y, color="orange", alpha=0.2)
axs[1, 0].set_title("Lebesgue Measure Set PDF (Normalized Indicator)")
axs[1, 0].legend()
axs[1, 0].grid(True)

# Plot 4: Analogous CDF (Cumulative Measure / Total Measure)
cdf_y = []
for x in x_domain:
    # Measure of E intersecting (-inf, x]
    cum_measure = 0.0
    for start, end in input_set:
        if x > start:
            cum_measure += min(x, end) - start
    cdf_y.append(cum_measure / measure_E)

axs[1, 1].plot(x_domain, cdf_y, color="green", lw=2, label="CDF")
axs[1, 1].set_title("Lebesgue Measure Set CDF")
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()