# complex_roots_app.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from math import sin, cos, atan2, pi

def cis(angle): 
    return cos(angle) + sin(angle) * 1j

def comp_solution(real, imaginary, root):
    r = (real**2 + imaginary**2)**0.5
    theta = atan2(imaginary, real)
    return [r**(1/root) * cis((theta + 2 * pi * k) / root) for k in range(root)]

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

# Streamlit UI
st.title("Complex Root Visualizer")

real = st.number_input("Real part of complex number", value=-8.0)
imag = st.number_input("Imaginary part of complex number", value=0.0)
n = st.number_input("Number of roots", min_value=1, max_value=24, value=6, step=1)

roots = comp_solution(real, imag, n)
fig = plot_complex_solutions(roots)

st.pyplot(fig)
