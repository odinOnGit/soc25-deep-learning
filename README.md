# SoC'25 - Deep Learning

This repository contains my weekly learning progress and practice exercises for the **SoC'25 - Deep Learning** track.

---

## Week 1: Python, Matplotlib & NumPy Basics

- Practiced core Python syntax and functions
- Created basic and advanced **matplotlib** plots (scatter, sine, etc.)
- Loaded and displayed images using **Pillow**
- Learned and applied fundamental **NumPy** operations:
  - Array creation and manipulation
  - Broadcasting
  - Reshaping arrays
  - Vectorized mathematical functions (like sigmoid)

---

## Week 2: Forward & Backward Propagation

- Preprocessed datasets (e.g., CSV handling)
- Implemented **parameter initialization** (weights, biases)
- Built simple neural network logic using:
  - **Forward propagation**
  - **Cost calculation**
  - **Backward propagation**
  - **Parameter updates**

---

## Week 3: Neural Network Implementation

- Solved practice problems to reinforce understanding of:
  - Sigmoid & derivative functions
  - Weight updates using gradients
  - Accuracy computation
- Worked toward building a basic 2-layer neural network from scratch

---

## Week 4: Mini Project

- Applied all the above concepts into a small project
- Trained a neural network using NumPy
- Evaluated model performance on sample data

---

## Tools & Libraries Used

- Python 3
- NumPy
- Matplotlib
- Pillow

---

# 🦜 CUB-200-2011 Bird Species Classifier

> A deep learning project built as part of **Season of Code 2025** under the mentorship of **Jay Rathod**, with co-mentors **Aditya** and **Noorain**.

![Confusion Matrix](outputs/plots/confusion_matrix.png)

## 📌 Project Overview

This project involves training a **MobileNetV2** model on the **Caltech-UCSD Birds-200-2011 (CUB-200-2011)** dataset, which consists of 11,788 images of 200 bird species. The goal is to classify bird species based on images using transfer learning and evaluate the model’s performance.

---

## 📂 Directory Structure

```bash
├── data/                       # CUB_200_2011 dataset (not pushed due to size)
├── outputs/
│   ├── plots/                  # Evaluation plots (e.g., confusion matrix)
│   └── mobilenet_cub.pth      # Best model weights (saved after training)
├── report/                    # LaTeX report (Overleaf-compatible)
├── src/
│   ├── dataset.py             # Custom PyTorch dataset
│   ├── model.py               # Model definition (MobileNetV2)
│   ├── train.py               # Training loop
│   └── evaluate.py            # Evaluation & metrics generation                 
└── README.md          # Readme file


