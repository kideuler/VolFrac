from sympy import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib import cm

# Define the symbols
t = symbols('t')

# Function to plot a parametric curve colored by curvature
def plot_with_curvature_color(x_expr, y_expr, curvature_expr, t_range, title="", **kwargs):
    t_min, t_max = t_range
    t_vals = np.linspace(t_min, t_max, 1000)
    
    # Convert sympy expressions to numerical functions
    x_func = lambdify(t, x_expr)
    y_func = lambdify(t, y_expr)
    curvature_func = lambdify(t, curvature_expr)
    
    # Evaluate at all t points
    x_vals = x_func(t_vals)
    y_vals = y_func(t_vals)
    curv_vals = curvature_func(t_vals)
    
    # Create a figure
    plt.figure(figsize=(10, 8))
    
    # Plot the curve with curvature coloring
    norm = Normalize(vmin=min(curv_vals), vmax=max(curv_vals))
    points = np.array([x_vals, y_vals]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Create a line collection
    lc = plt.matplotlib.collections.LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(np.array(curv_vals))
    lc.set_linewidth(3)
    
    # Add the colored line to the plot
    ax = plt.gca()
    ax.add_collection(lc)
    
    # Add a colorbar
    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_label('Curvature')
    
    # Set limits and aspect ratio
    if 'xlim' in kwargs:
        plt.xlim(kwargs['xlim'])
    if 'ylim' in kwargs:
        plt.ylim(kwargs['ylim'])
    if 'aspect_ratio' in kwargs:
        plt.gca().set_aspect(kwargs['aspect_ratio'][1]/kwargs['aspect_ratio'][0])
    else:
        plt.gca().set_aspect('equal')
    
    plt.grid(True)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# Keep your existing calculations for all three shapes, but replace the plotting sections

# Original calculations for ellipse
# Ellipse
x = 0.2*cos(t) + 0.5
y = 0.1*sin(t) + 0.5

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

# Compute curvature
ellipse_curvature = (x_prime*y_double_prime - y_prime*x_double_prime)/(x_prime**2 + y_prime**2)**(3/2)
print("ellipse curvature = ", ellipse_curvature)

# Plot ellipse with curvature
plot_with_curvature_color(x, y, ellipse_curvature, (0.0, 6.28), 
                          "Ellipse with Curvature", 
                          xlim=(0.25,0.75), ylim=(0.25,0.75), 
                          aspect_ratio=(1, 1))

area = integrate(y*x_prime, (t, 0, 2*pi))
print("ellipse area = ", area)


# Flower
x = (0.25 + 0.1*sin(5*t))*cos(t)/3.0 + 0.5
y = (0.25 + 0.1*sin(5*t))*sin(t)/3.0 + 0.5

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

# compute curvature
flower_curvature = (x_prime*y_double_prime - y_prime*x_double_prime)/(x_prime**2 + y_prime**2)**(3/2)
print("flower curvature = ", flower_curvature)

# Plot flower with curvature
plot_with_curvature_color(x, y, flower_curvature, (0.0, 6.28), 
                          "Flower with Curvature", 
                          xlim=(0.25,0.75), ylim=(0.25,0.75), 
                          aspect_ratio=(1, 1))

area = integrate(y*x_prime, (t, 0, 2*pi))
print("flower area = ", area)


# Extreme Petal
x = (0.25 + 0.15*sin(20*t))*cos(t)/3.0 + 0.5
y = (0.25 + 0.1*sin(20*t))*sin(t)/3.0 + 0.5

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

# Compute curvature
extreme_curvature = (x_prime*y_double_prime - y_prime*x_double_prime)/(x_prime**2 + y_prime**2)**(3/2)
print("extreme petal curvature = ", extreme_curvature)

# Plot extreme petal with curvature
plot_with_curvature_color(x, y, extreme_curvature, (0.0, 6.28), 
                          "Extreme Petal with Curvature", 
                          xlim=(0.25,0.75), ylim=(0.25,0.75), 
                          aspect_ratio=(1, 1))

area = integrate(y*x_prime, (t, 0, 2*pi))
print("extreme petal area = ", area)