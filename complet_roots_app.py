# complex_roots_app.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from math import sin, cos, atan2, pi

# ----------------------------
# Math functions
# ----------------------------
def cis(angle): 
    return cos(angle) + sin(angle) * 1j

def comp_solution(real, imaginary, root, angle_offset=0):
    r = (real**2 + imaginary**2)**0.5
    theta = atan2(imaginary, real)
    return [r**(1/root) * cis(((theta + 2 * pi * k) / root) + angle_offset) for k in range(root)]

# ----------------------------
# Plotting function
# ----------------------------
def plot_complex_solutions(complex_nums):
    real_parts = [z.real for z in complex_nums]
    imag_parts = [z.imag for z in complex_nums]
    radius = abs(complex_nums[0])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axhline(0, color='black', linewidth=1.2)
    ax.axvline(0, color='black', linewidth=1.2)
    ax.scatter(real_parts, imag_parts, color='blue')

    theta = np.linspace(0, 2 * np.pi, 500)
    ax.plot(radius * np.cos(theta), radius * np.sin(theta), '--', color='green', label=f'|z| = {round(radius, 2)}')

    for i, z in enumerate(complex_nums):
        label = f"z{i} = {round(z.real, 2)} + {round(z.imag, 2)}j"
        ax.text(z.real, z.imag, label, fontsize=9, ha='left', va='bottom')

    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title("Complex Roots")
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.legend()
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

# Optional: animated rotation
angle_offset = st.slider("Animate rotation (radians)", 0.0, 2 * pi, step=0.1)

# Explanation panel
with st.expander("ℹ️ About this app"):
    st.markdown("""
    This app plots the **n complex roots** of a given complex number using **De Moivre's Theorem**.

    - Roots are evenly spaced around a circle
    - You can input in rectangular or polar form
    - The angle offset slider rotates the entire root set
    
    Complex roots are given by:
    \[ z_k = r^{1/n} \text{cis}\left( \frac{\theta + 2\pi k}{n} \right) \]
    """)

# Compute and plot
roots = comp_solution(real, imag, n, angle_offset)
fig = plot_complex_solutions(roots)

st.pyplot(fig)
