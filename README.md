# Heat Inverse PINN Solver

This repository contains the implementation of a Physics-Informed Neural Network (PINN) designed to solve the 1D Heat Equation.

## Project Overview
This project explores using Deep Learning to solve partial differential equations (PDEs) in the context of heat transfer:

- **Forward Modeling:** An interactive web application built with **Streamlit** that predicts temperature fields $u(x, t)$ based on spatial and temporal inputs.
- **Inverse Modeling:** The core logic for identifying the physical parameter $\alpha$ (thermal diffusivity) is implemented in `heat_inverse.py`. 

*Note: Due to resource limitations on serverless cloud environments (Streamlit Cloud), the live web app currently hosts the Forward Solver. The full Inverse Solver implementation, including the training and optimization logic, is available in the repository files.*

## Tech Stack
* **Modeling:** [DeepXDE](https://deepxde.readthedocs.io/), TensorFlow
* **Web Interface:** Streamlit
* **Environment:** Python 3.12 

## Repository Structure
- `app.py`: The live Streamlit application (Forward Solver).
- `heat_inverse.py`: The complete script for Inverse Parameter Estimation (identifying $\alpha$).
- `heat_model_weights-717.weights.h5`: Pre-trained model weights.
- `requirements.txt`: Project dependencies.

## About
Developed by Lourdus Rogin Melkrin S( Pursuing Chemical Engineering at IIT (ISM) Dhanbad)
