from sympy import *
import matplotlib.pyplot as plt
import numpy as np

# Define the symbols
t = symbols('t')

# Define the x coordinate as a piecewise function
x = Piecewise(
    (0.5*cos(pi*t), And(t >= 0.0, t < 1.0)),
    (-0.5*(2-t)**2 - 0.5*(4-2*t)*(t-1) - 0.25*(t-1)**2, And(t >= 1.0, t < 2.0)),
    (1.25*t - 2.75, And(t >= 2.0, t < 3.0)),
    ((4-t)**2 + 0.5*(8-2*t)*(t-3) + 0.5*(t-3)**2, And(t >= 3.0, t < 4.0)),
    (0.1*cos(pi*(2*t-8)) - 0.2, And(t >= 4.0, t < 5.0)),
    (0.1*cos(pi*(2*t-8)) + 0.2, And(t >= 5.0, t < 6.0))
)

# Define the y coordinate as a piecewise function
y = Piecewise(
    (0.5*sin(pi*t) + 1.0, And(t >= 0.0, t < 1.0)),
    ((1.0*(2-t))**2 + 0.5*(4-2*t)*(t-1), And(t >= 1.0, t < 2.0)),
    (0.05*sin(pi*(6.0*t-12.0)), And(t >= 2.0, t < 3.0)),
    (0.5*(8-2*t)*(t-3) + 1.0*(t-3)**2, And(t >= 3.0, t < 4.0)),
    (0.2*sin(pi*(2*t-8)) + 1.0, And(t >= 4.0, t < 5.0)),
    (0.2*sin(pi*(2*t-8)) + 1.0, And(t >= 5.0, t < 6.0))
)

# Define the dx/dt derivative
dx_dt = diff(x, t)

# Define the dy/dt derivative
dy_dt = diff(y, t)

area = integrate(y*dx_dt, (t, 0, 4))
area_eye = integrate(y*dx_dt, (t, 4, 5))
print("area = ", area - 2*area_eye)

# Calculate second derivatives for curvature calculation
d2x_dt2 = diff(dx_dt, t)
d2y_dt2 = diff(dy_dt, t)

# Display the functions
print("data[0] = ",ccode(simplify(x)),";")
print("\ndata[1] = ",ccode(simplify(y)),";")
print("\ndata[2] = ",ccode(simplify(dx_dt)),";")
print("\ndata[3] = ",ccode(simplify(dy_dt)),";")
print("\ndata[4] = ",ccode(simplify(d2x_dt2)),";")
print("\ndata[5] = ",ccode(simplify(d2y_dt2)),";")

# Plot the ghost shape
def plot_ghost():
    t_vals = np.linspace(0, 6, 1000)
    x_vals = []
    y_vals = []
    
    # Main body
    for t_val in t_vals:
        if t_val < 4.0:
            x_val = float(x.subs([(t, t_val), (pi, np.pi)]))
            y_val = float(y.subs([(t, t_val), (pi, np.pi)]))
            x_vals.append(x_val)
            y_vals.append(y_val)
    
    # Plot main body
    plt.figure(figsize=(10, 8))
    plt.plot(x_vals, y_vals, 'b-', label='Ghost Body')
    
    # Left eye (t from 4 to 5)
    x_left_eye = []
    y_left_eye = []
    for t_val in np.linspace(4, 5-0.001, 200):
        x_val = float(x.subs([(t, t_val), (pi, np.pi)]))
        y_val = float(y.subs([(t, t_val), (pi, np.pi)]))
        x_left_eye.append(x_val)
        y_left_eye.append(y_val)
    plt.plot(x_left_eye, y_left_eye, 'r-', label='Left Eye')
    
    # Right eye (t from 5 to 6)
    x_right_eye = []
    y_right_eye = []
    for t_val in np.linspace(5, 6, 200):
        x_val = float(x.subs([(t, t_val), (pi, np.pi)]))
        y_val = float(y.subs([(t, t_val), (pi, np.pi)]))
        x_right_eye.append(x_val)
        y_right_eye.append(y_val)
    plt.plot(x_right_eye, y_right_eye, 'g-', label='Right Eye')
    
    plt.axis('equal')
    plt.grid(True)
    #plt.title('Ghost Shape')
    plt.legend(fontsize=16)
    plt.draw()
    # Make sure it gets displayed by flushing events
    plt.pause(0.5)

# Call the plotting function
plot_ghost()

plt.savefig('build/results/Ghost.png', dpi=300, bbox_inches='tight')