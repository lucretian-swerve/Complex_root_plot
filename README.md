# Complex Root Visualizer

This interactive web app visualizes the **n complex roots** of a given complex number using **De Moivre's Theorem**. It supports both rectangular and polar input modes and provides intuitive graphical and algebraic explanations.

![screenshot](path_to_screenshot.png) <!-- Replace with an actual screenshot path or Streamlit link -->

## Features

- Input complex numbers in **rectangular (a + bi)** or **polar (r ∠ θ)** form
- Toggle angle units between **degrees** and **radians**
- Specify the number of roots (up to 24)
- View the **root equation** and **plot of all roots**
- Interactive **root verification**: 
  - Raises selected root to the power n
  - Compares with the original number
  - Shows the **binomial expansion** with symbolic and numeric steps

## How It Works

This app is powered by [Streamlit](https://streamlit.io/) and uses:

- `NumPy` and `matplotlib` for numerical computation and plotting
- `SymPy` for symbolic math (binomial expansion and simplification)
- De Moivre's formula:  
  \[
  z_k = r^{1/n} \cdot \text{cis}\left( \frac{\theta + 2\pi k}{n} \right)
  \]

## Try the App

 **Live Demo**: [your-app-name.streamlit.app](https://your-app-name.streamlit.app)  
(Replace with your actual Streamlit Cloud link.)

## Installation (Optional)

To run locally:

```bash
git clone https://github.com/yourusername/complex-root-visualizer.git
cd complex-root-visualizer
pip install -r requirements.txt
streamlit run complex_roots_app.py
