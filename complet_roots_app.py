# complex_roots_app.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from math import sin, cos, atan2, pi, e
from sympy import symbols, I, expand, simplify, collect, latex


# ----------------------------
# Math functions
# ----------------------------
def cis(angle): 
    return cos(angle) + sin(angle) * 1j

def comp_solution(real, imaginary, root):
    r = (real**2 + imaginary**2)**0.5
    theta = atan2(imaginary, real)
    return [r**(1/root) * cis((theta + 2 * pi * k) / root) for k in range(root)]

# ----------------------------
# Plotting function
# ----------------------------
def plot_complex_solutions(complex_nums, fixed_limit=None):
    real_parts = [z.real for z in complex_nums]
    imag_parts = [z.imag for z in complex_nums]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axhline(0, color='black', linewidth=1.2)
    ax.axvline(0, color='black', linewidth=1.2)
    ax.scatter(real_parts, imag_parts, color='blue')

    radius = max(abs(z) for z in complex_nums)
    theta = np.linspace(0, 2 * np.pi, 500)
    ax.plot(radius * np.cos(theta), radius * np.sin(theta), '--', color='green')

    for i, z in enumerate(complex_nums):
        label = f"z{i} = {round(z.real, 2)} + {round(z.imag, 2)}j"
        ax.text(z.real, z.imag, label, fontsize=9, ha='left', va='bottom')

    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title("Complex Roots")
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")

    if fixed_limit:
        ax.set_xlim(-fixed_limit, fixed_limit)
        ax.set_ylim(-fixed_limit, fixed_limit)
    else:
        lim = max(max(abs(np.array(real_parts))), max(abs(np.array(imag_parts)))) * 1.1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    return fig

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🔢 Complex Root Visualizer")

# Input mode toggle
mode = st.radio("Input Mode", ["Rectangular (a + bi)", "Polar (r ∠ θ)"])

# Input fields
if mode == "Rectangular (a + bi)":
    real = st.number_input("Real part (a)", value=1.0)
    imag = st.number_input("Imaginary part (b)", value=0.0)
else:
    r = st.number_input("Modulus (r)", value=1.0, min_value=0.0)
    angle_mode = st.radio("Angle input mode", ["Degrees", "Radians"])
    if angle_mode == "Degrees":
        theta_deg = st.number_input("Angle θ (degrees)", value=180.0)
        theta = np.deg2rad(theta_deg)
        st.caption(f"θ = {round(theta, 4)} radians")
    else:
        user_input = st.text_input("Angle θ (radians)", value="pi")
        try:
            # Allow expressions like "pi", "pi/2", "3*pi/4"
            theta = eval(user_input, {"__builtins__": None}, {"pi": pi, "e": e})
            st.caption(f"θ ≈ {round(theta, 4)} radians")
        except:
            st.error("Invalid expression. Try something like `pi/2` or `3*pi/4`.")
            theta = 0.0  # default fallback
    real = r * np.cos(theta)
    imag = r * np.sin(theta)
    st.caption(f"Converted to rectangular: a = {round(real, 2)}, b = {round(imag, 2)}")

n = st.number_input("Number of roots (n)", min_value=1, max_value=24, value=3, step=1)

# Show root equation
if imag < 0:
    eq = f"x^{n} = {real} - {abs(imag)}i"
else:
    eq = f"x^{n} = {real} + {imag}i"
st.markdown(f"### Root Equation:")
st.latex(f"x^{n} = {round(real, 2)} {'-' if imag < 0 else '+'} {round(abs(imag), 2)}i")


# Explanation panel
with st.expander("ℹ️ About this app"):
    st.markdown("""
    This app plots the **n complex roots** of a given complex number using **De Moivre's Theorem**.
    """)
    st.latex(r"z_k = r^{1/n} \cdot \text{cis}\left( \frac{\theta + 2\pi k}{n} \right)")


# Compute and display roots
roots = comp_solution(real, imag, n)
max_radius = max(abs(z) for z in roots) * 1.1
fig = plot_complex_solutions(roots, fixed_limit=max_radius)
st.pyplot(fig)


# ----------------------------
# Root verification section
# ----------------------------
st.subheader("🧪 Root Verification")

# Select a root to verify
root_index = st.slider("Select root to verify (zₖ)", min_value=0, max_value=n-1, value=0, step=1)
zk = roots[root_index]
zk_powered = zk ** n
zk_powered_clean = complex(round(zk_powered.real, 4), round(zk_powered.imag, 4))
original_clean = complex(round(real, 4), round(imag, 4))
error = abs(zk_powered - (real + imag * 1j))

# Display selected root
st.markdown(f"### Selected Root: $z_{{{root_index}}}$")
st.latex(rf"z_{{{root_index}}} = {round(zk.real, 4)} {'-' if zk.imag < 0 else '+'} {round(abs(zk.imag), 4)}i")

# Show expression raised to power
st.markdown("### Raised to the Power:")
st.latex(rf"z_{{{root_index}}}^{{{n}}} = ({round(zk.real, 4)} {'-' if zk.imag < 0 else '+'} {round(abs(zk.imag), 4)}i)^{{{n}}}")

# Binomial expansion (only if n is small)
if n <= 6:
    st.markdown("### 🔍 Binomial Expansion:")

    # Use full-precision a and b from the selected root
    a_val = float(zk.real)
    b_val = float(zk.imag)

    # Declare symbolic variables
    a, b = symbols('a b')
    z_sym = a + b * I

    # Perform symbolic expansion and simplification
    expanded = expand(z_sym ** n)
    collected = collect(expanded, I)
    substituted_expr = simplify(expanded.subs({a: a_val, b: b_val}))

    # Evaluate real and imaginary parts
    real_part, imag_part = substituted_expr.as_real_imag()
    re_val = real_part.evalf()
    im_val = imag_part.evalf()

    # Format to 2 decimal places for clean output
    re_str = format(re_val, ".1f")
    im_str = format(abs(im_val), ".1f")
    sign = "-" if im_val < 0 else "+"

    # Display steps
    st.latex(rf"(a + bi)^{{{n}}} = {latex(expanded)}")
    st.latex(rf"= {latex(collected)}")
    st.latex(rf"\text{{where }} a = {round(a_val, 6)}, \quad b = {round(b_val, 6)}")
    st.latex(rf"= {re_str} {sign} {im_str}i")


# Show result
st.markdown("### Result:")
st.latex(rf"z_{{{root_index}}}^{{{n}}} = {zk_powered_clean.real} {'-' if zk_powered_clean.imag < 0 else '+'} {abs(zk_powered_clean.imag)}i")

# Show original
st.markdown("### Original Equation:")
st.latex(rf"x^{{{n}}} = {original_clean.real} {'-' if original_clean.imag < 0 else '+'} {abs(original_clean.imag)}i")
