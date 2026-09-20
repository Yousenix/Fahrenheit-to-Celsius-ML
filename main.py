import numpy as np


spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

x = np.array([10, 50, 0])
y = np.array([-12.22, 10, -17.78])

np.set_printoptions(precision=2)

w = 0
b = 0

learning_rate = 0.00001

print("""
╔══════════════════════════════╗
║     Linear Regression ML     ║
╚══════════════════════════════╝
""")

def model(inp):

    global test

    test = inp

    try:
        w, b = loadmodel()
        print("Model loaded ✓")

    except FileNotFoundError:
        w, b = creatmodel()
        

    y_pred = w * x + b
    test_pred = w * test + b
    print(f"Celsius : {test_pred[0]:.2f}")

def loadmodel():

    model = np.load("fahrenheit_model.npz")

    w = model["w"]
    b = model["b"]

    return w, b

def creatmodel():

    global w, b

    for i in range(10000000):

        y_pred = w * x + b
        error = y_pred - y

        gradient_w = (2 / len(x)) * np.sum(x * error)
        gradient_b = (2 / len(x)) * np.sum(error)

        w = w - learning_rate * gradient_w
        b = b - learning_rate * gradient_b

        if i % 100000 == 0:
            print(f"\rLearning {spinner[(i // 100000) % len(spinner)]} ", end="", flush=True)

    print("\rTraining complete! ✓")

    np.savez("fahrenheit_model.npz", w=w, b=b)

    return w, b

model(np.array([int(input("Fahrenheit : "))]))