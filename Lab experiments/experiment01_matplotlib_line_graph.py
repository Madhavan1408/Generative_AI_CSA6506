"""Experiment 1: Install Matplotlib and display a simple line graph."""
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [10, 20, 15, 30, 25]

plt.plot(x, y, marker="o")
plt.title("Simple Line Graph")
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.grid(True)
plt.tight_layout()
plt.show()
