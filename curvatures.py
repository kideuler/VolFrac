from sympy import *

# Define the symbols
t = symbols('t')

# Ellipse
x = 0.2*cos(t) + 0.5
y = 0.1*sin(t) + 0.5

plot_parametric(x, y, (t, -pi, pi),aspect_ratio=(1, 1),xlim=(0.25,0.75), ylim=(0.25,0.75))

# Define the first derivatives
x_prime = diff(x, t)
y_prime = diff(y, t)
print("x' = ", x_prime)
print("y' = ", y_prime)

# Define the second derivatives
x_double_prime = diff(x_prime, t)
y_double_prime = diff(y_prime, t)
print("x'' = ", x_double_prime)
print("y'' = ", y_double_prime)

area = integrate(y*x_prime, (t, 0, 2*pi))
print("area = ",area)


# Flower
x = (0.25 + 0.1*sin(5*t))*cos(t)/3.0 + 0.5
y = (0.25 + 0.1*sin(5*t))*sin(t)/3.0 + 0.5

plot_parametric(x, y, (t, -pi, pi),aspect_ratio=(1, 1),xlim=(0.25,0.75), ylim=(0.25,0.75))

# Define the first derivatives
x_prime = diff(x, t)
y_prime = diff(y, t)
print("x' = ", x_prime)
print("y' = ", y_prime)

# Define the second derivatives
x_double_prime = diff(x_prime, t)
y_double_prime = diff(y_prime, t)
print("x'' = ", x_double_prime)
print("y'' = ", y_double_prime)

area = integrate(y*x_prime, (t, 0, 2*pi))
print("area = ",area)


# Extreme Petal
x = (0.25 + 0.15*sin(20*t))*cos(t)/3.0 + 0.5
y = (0.25 + 0.1*sin(20*t))*sin(t)/3.0 + 0.5

plot_parametric(x, y, (t, -pi, pi),aspect_ratio=(1, 1),xlim=(0.25,0.75), ylim=(0.25,0.75))

# Define the first derivatives
x_prime = diff(x, t)
y_prime = diff(y, t)
print("x' = ", x_prime)
print("y' = ", y_prime)

# Define the second derivatives
x_double_prime = diff(x_prime, t)
y_double_prime = diff(y_prime, t)
print("x'' = ", x_double_prime)
print("y'' = ", y_double_prime)

area = integrate(y*x_prime, (t, 0, 2*pi))
print("area = ",area)
