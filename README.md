# Fahrenheit → Celsius ML Model 🌡️

A simple **Machine Learning model** built from scratch with **Python and NumPy** to predict Celsius values from Fahrenheit temperatures.

This project demonstrates the basic workflow of a machine learning model:

**Training → Learning Parameters → Saving the Model → Loading the Model → Prediction**

## 🧠 About the Model

The model uses **Linear Regression** and learns two parameters during training:

* `w` — weight
* `b` — bias

Instead of directly implementing the Fahrenheit-to-Celsius formula, the model **learns the relationship from training data using Gradient Descent**.

After training, the learned parameters are saved to:

```text
fahrenheit_model.npz
```

This allows the trained model to be loaded and used for predictions without training again.

## ⚙️ How It Works

The model starts with:

```text
w = 0
b = 0
```

It then repeatedly:

1. Calculates predictions.
2. Calculates the error.
3. Calculates the gradients.
4. Updates `w` and `b`.
5. Saves the learned parameters.

The prediction is then calculated from the learned model.

## 🚀 Usage

Enter the Fahrenheit temperature in Terrminal.

For example:

```Shell
Fahrenheit : 100
```

The model will return the predicted Celsius values.

Example:

```text
Celsius : 37.78
```

## 💾 Model Persistence

After the model is trained, its parameters are stored using NumPy:

```python
np.savez("fahrenheit_model.npz", w=w, b=b)
```

When the program runs again, it loads the existing model instead of training from scratch.

## 🛠️ Built With

* Python
* NumPy
* Linear Regression
* Gradient Descent

## 📚 Purpose

This is a small educational project created to understand the fundamentals of **Machine Learning**, especially how a model can learn parameters from data instead of relying on a manually written formula.

> This project is intentionally simple and implements the learning process from scratch rather than using a machine-learning library such as scikit-learn.
