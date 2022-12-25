import numpy as np
from math import pi, sin, cos

'''
# Rotate point using phase
def rotate_point(curr_pt, next_pt, phase):
    return curr_pt*sin(phase) + next_pt*cos(phase)

# Rotate on x-axis 
# https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Rotation_of_coordinates.svg/320px-Rotation_of_coordinates.svg.png
def rotate_x(x, y, z, phase):
    y_new = rotate_point(y, -1*z, phase)
    z_new = rotate_point(z, y, phase)

    return x, y_new, z_new


# Rotate on y-axis
def rotate_y(x, y, z, phase):
    x_new = rotate_point(x, z, phase)
    z_new = rotate_point(z, -1*x, phase)

    return x_new, y, z_new

# Rotate on z-axis
def rotate_z(x, y, z, phase):
    x_new = rotate_point(x, -1*y, phase)
    y_new = rotate_point(y, x, phase)

    return x_new, y_new, z
'''


# Improved method for rotation
degree = 1

# Convert degree to radians
theta = degree*pi/180

# Define rotation matrix constants
z_rotation_matrix = np.array([ 
        [ cos(theta), -sin(theta), 0 ], 
        [ sin(theta), cos(theta), 0 ], 
        [ 0, 0, 1 ]
    ])

x_rotation_matrix = np.array([ 
        [ 1, 0, 0 ], 
        [ 0, cos(theta), -sin(theta) ], 
        [ 0, sin(theta), cos(theta) ]
    ])

y_rotation_matrix = np.array([ 
        [ cos(theta), 0, sin(theta) ], 
        [ 0, 1, 0], 
        [ -sin(theta), 0, cos(theta)]
    ])

# Perform matrix multiplication

# TODO:
# rotation_matrix * [[x], [y], [z]]
# [x, y, z] * rotation_matrix
