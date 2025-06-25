# complex_roots_app.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from math import sin, cos, atan2, pi, e
from sympy import symbols, I, expand, simplify, collect, latex
import pandas as pd

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
def plot_complex_solutions(complex_nums, fixed_limit=None, connect=False):
    real_parts = [z.real for z in complex_nums]
    imag_parts = [z.imag for z in complex_nums]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axhline(0, color='black', linewidth=1.2)
    ax.axvline(0, color='black', linewidth=1.2)
    ax.scatter(real_parts, imag_parts, color='blue')

    # Draw circle
    radius = max(abs(z) for z in complex_nums)
    theta = np.linspace(0, 2 * np.pi, 500)
    ax.plot(radius * np.cos(theta), radius * np.sin(theta), '--', color='green')

    # Connect roots if enabled
    if connect and len(complex_nums) > 1:
        for i in range(len(complex_nums)):
            z1 = complex_nums[i]
            z2 = complex_nums[(i + 1) % len(complex_nums)]
            ax.plot([z1.real, z2.real], [z1.imag, z2.imag], color='blue', linestyle='-', linewidth=2.5)

    # Annotate roots
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
st.title("Complex Root Visualizer")

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
            theta = eval(user_input, {"__builtins__": None}, {"pi": pi, "e": e})
            st.caption(f"θ ≈ {round(theta, 4)} radians")
        except:
            st.error("Invalid expression. Try something like `pi/2` or `3*pi/4`.")
            theta = 0.0
    real = r * np.cos(theta)
    imag = r * np.sin(theta)
    st.caption(f"Converted to rectangular: a = {round(real, 2)}, b = {round(imag, 2)}")

n = st.number_input("Number of roots (n)", min_value=1, max_value=24, value=3, step=1)

# Show root equation
st.markdown("### Root Equation:")
st.latex(f"x^{n} = {round(real, 2)} {'-' if imag < 0 else '+'} {round(abs(imag), 2)}i")

# Root calculation breakdown
st.markdown("### Root Calculations")

r_input = (real**2 + imag**2)**0.5
theta_input = atan2(imag, real)

for k in range(n):
    angle_k = (theta_input + 2 * pi * k) / n
    root_mod = r_input ** (1/n)
    z_k = root_mod * cis(angle_k)
    re = round(z_k.real, 2)
    im = round(z_k.imag, 2)
    st.latex(
        rf"z_{{{k}}} = \sqrt[{n}]{{{round(r_input, 2)}}} \cdot \text{{cis}}\left( \frac{{\theta + 2\pi \cdot {k}}}{{{n}}} \right) = "
        rf"{round(root_mod, 2)} \cdot \text{{cis}}({round(angle_k, 2)}\ \text{{rad}}) = {re} {'-' if im < 0 else '+'} {abs(im)}i"
    )

# Compute roots
roots = comp_solution(real, imag, n)
max_radius = max(abs(z) for z in roots) * 1.1

# Symmetry toggle
connect_roots = st.checkbox("Show symmetry lines (connect roots)")
fig = plot_complex_solutions(roots, fixed_limit=max_radius, connect=connect_roots)
st.pyplot(fig)

# Explanation panel
with st.expander("Why do roots form a regular polygon?"):
    st.markdown("""
    When a complex number is raised to the \( n \)th power, its roots are:
    
    - Equally spaced around a circle in the complex plane  
    - Separated by $\\frac{2\\pi}{n}$ radians  
    - Located at the vertices of a regular $n$-gon
    """)

    st.markdown("This symmetry arises from **De Moivre's Theorem**, which places each root at:")
    st.latex(r"z_k = r^{1/n} \cdot \text{cis} \left( \frac{\theta + 2\pi k}{n} \right)")
    st.markdown("""
    When the base number (r) is 1, these roots lie **on the unit circle** and are called the **roots of unity**.
    """)



# Root table
with st.expander("View Roots as a Table"):
    data = [
        {
            "zₖ": f"z{k}",
            "Rectangular": f"{round(z.real, 2)} {'-' if z.imag < 0 else '+'} {round(abs(z.imag), 2)}i",
            "Modulus": round(abs(z), 2),
            "Angle (deg)": round(np.degrees(np.angle(z)), 2)
        } for k, z in enumerate(roots)
    ]
    df = pd.DataFrame(data)
    st.dataframe(df)

# About section
with st.expander("About this app"):
    st.markdown("""
    This app plots the **n complex roots** of a given complex number using **De Moivre's Theorem**.
    """)
    st.latex(r"z_k = r^{1/n} \cdot \text{cis}\left( \frac{\theta + 2\pi k}{n} \right)")
