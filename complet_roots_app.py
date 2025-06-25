# complex_roots_app.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time
from math import sin, cos, atan2, pi

# ----------------------------
# Math functions
# ----------------------------
def cis(angle): 
    return cos(angle) + sin(angle) * 1j

def comp_solution(real, imaginary, root):
    r = (real**2 + imaginary**2)**0.5
    theta = atan2(imaginary, real)
    return [r**(1/root) * cis((theta + 2 * pi * k) / root) for k in range(root)]

def rotate_and_converge(roots, power=1):
    return [abs(z)**power * cis((np.angle(z) % (2 * pi)) * power) for z in roots]

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

    # Lock axis limits to prevent plot from resizing
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
    real = st.number_input("Real part (a)", value=-8.0)
    imag = st.number_input("Imaginary part (b)", value=0.0)
else:
    r = st.number_input("Modulus (r)", value=8.0, min_value=0.0)
    theta_deg = st.number_input("Angle θ (degrees)", value=180.0)
    theta = np.deg2rad(theta_deg)
    real = r * np.cos(theta)
    imag = r * np.sin(theta)
    st.caption(f"Converted to rectangular: a = {round(real, 2)}, b = {round(imag, 2)}")

n = st.number_input("Number of roots (n)", min_value=1, max_value=24, value=6, step=1)
speed = st.slider("Animation Speed (lower = faster)", min_value=1, max_value=100, value=20, step=1)

# Explanation panel
with st.expander("ℹ️ About this app"):
    st.markdown("""
    This app plots the **n complex roots** of a given complex number using **De Moivre's Theorem**.

    - Roots are evenly spaced around a circle
    - You can input in rectangular or polar form
    - Click the button below to animate the roots converging to the original number (via exponentiation)

    Complex roots are given by:
    \[ z_k = r^{1/n} \text{cis}\left( \frac{\theta + 2\pi k}{n} \right) \]
    """)

# Animate or reset
animate = st.button("▶ Animate Convergence")
reset = st.button("⏹ Reset")

placeholder = st.empty()
roots = comp_solution(real, imag, n)
max_radius = max(abs(z) for z in roots) ** n * 1.1

if animate:
    for exp in np.linspace(1, n, 100):
        if reset:
            break
        converged = rotate_and_converge(roots, power=exp)
        fig = plot_complex_solutions(converged, fixed_limit=max_radius)
        placeholder.pyplot(fig)
        time.sleep(speed / 10000.0)  # Smoother and faster max animation
else:
    fig = plot_complex_solutions(roots, fixed_limit=max_radius)
    placeholder.pyplot(fig)
