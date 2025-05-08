# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:28:18 2024

@author: buchmiller
"""
import math
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt

# Define parameters for the equation
R_0 = 100  # Initial value of R
#R_t = 98.33  # Theoretical measurement value of R; for testing purposes
A = 3.9083 * 10**-3  # Coefficient A
B = -5.775 * 10**-7  # Coefficient B
C = -4.183 * 10**-12  # Coefficient C


def plot_results(R_t_values, newton_solutions, exact_solutions, difference):
    """
    Plots the results comparing Newton-Raphson and exact solutions, and the
    difference between them.
    
    Parameters:
    R_t_values (ndarray): Array of R_t values used for calculations.
    newton_solutions (list): Solutions obtained from the Newton-Raphson method.
    exact_solutions (list): Solutions obtained from the exact quadratic solution.
    difference (ndarray): Differences between Newton-Raphson and exact solutions.
    """
    
    # Plot the Newton-Raphson and exact solutions
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(R_t_values, newton_solutions, color='#CD0000', label="Newton-Raphson Solution")
    plt.plot(R_t_values, exact_solutions, color='#00008B', alpha=0.5, label="Exact Solution")
    plt.xlabel(r'$R_T \, \mathrm{in} \, \mathrm{\Omega}$')
    plt.ylabel("T in °C")
    #plt.title("Comparison of Newton-Raphson and Exact Solutions")
    plt.legend()
    plt.savefig('comparison.png', transparent=True)

    # Plot the differences between solutions
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(R_t_values, difference, color='#00008B', label="Difference (Newton-Raphson - Exact)")
    plt.ylim(-0.000001, 0.000001)
    plt.xlabel(r'$R_T \, \mathrm{in} \, \mathrm{\Omega}$')
    plt.ylabel(r'$T_{diff}$ in °C')
    #plt.title("Difference Between Newton-Raphson and Exact Solutions")
    plt.legend()
    plt.savefig('difference.png', transparent=True)


def solve_for_t(R_t, R_0, A, B):
    """
    Solves for the variable 't' in the equation:
    
        R_t = R_0 * (A * t + B * t^2 + 1)
    
    by rearranging it into a standard quadratic form:
    
        B * t^2 + A * t + (1 - R_t / R_0) = 0
    
    Parameters:
    R_t (float): The target value of R at time t.
    R_0 (float): The initial value of R at t = 0.
    A (float): Coefficient of the linear term in the quadratic equation.
    B (float): Coefficient of the quadratic term in the quadratic equation.
    
    Returns:
    float or None: The value of 't' that satisfies the equation if there is 
                   a real solution. If there are no real solutions, returns None.
    """
    
    # Calculate coefficients for the quadratic equation
    a = B
    b = A
    c = 1 - R_t / R_0
    
    # Calculate the discriminant
    discriminant = b**2 - 4 * a * c
    
    # Check if the discriminant is non-negative
    if discriminant < 0:
        return None  # No real solution
    
    # Compute the solutions for t
    sqrt_discriminant = math.sqrt(discriminant)
    t1 = (-b + sqrt_discriminant) / (2 * a)
    t2 = (-b - sqrt_discriminant) / (2 * a)
    
    # Return the larger of the two solutions (if real)
    return t1


def func_negative_t(t, R_0, R_t, A, B, C):
    """
    Defines the equation to solve for negative values of 't', which includes
    a quartic term. This function is used to find the root of the equation
    using the Newton-Raphson method.
    
    Parameters:
    t (float): The variable value for which we are solving.
    
    Returns:
    float: The result of the equation R_0 * (1 + A * t + B * t**2 + C * (t - 100) * t**3) - R_t.
    """
    return R_0 * (1 + A * t + B * t**2 + C * (t - 100) * t**3) - R_t


def func_positive_t(t, R_0, R_t, A, B, C):
    """
    Defines the equation to solve for positive values of 't'. 
    This function is also used to find the root of the equation
    using the Newton-Raphson method.
    
    Parameters:
    t (float): The variable value for which we are solving.
    
    Returns:
    float: The result of the equation R_0 * (1 + A * t + B * t**2) - R_t.
    """
    return R_0 * (1 + A * t + B * t**2) - R_t



def main(R_0, A, B, C):

    R_t_values = np.arange(100, 200.01, 0.01)  # Array of R_t values in increments of 0.01
    
    # Generate initial guesses for t based on differences between R_0 and R_t_values
    initial_guesses = abs(R_0 - R_t_values) * 3
    #print(initial_guesses)
    # Lists to store results from Newton-Raphson and exact solutions
    newton_solutions = []
    exact_solutions = []
    
    # Loop through each target R_t value and compute both Newton-Raphson and exact solutions
    for R_t, initial_guess in zip(R_t_values, initial_guesses):

        # Newton-Raphson solution for positive values of t
        newton_solution = opt.newton(func_positive_t, initial_guess, args=(R_0, R_t, A, B, C))
        
        # Exact solution using the quadratic formula
        exact_solution = solve_for_t(R_t, R_0, A, B)
        #print(newton_solution, exact_solution)
        newton_solutions.append(newton_solution)
        exact_solutions.append(exact_solution)
    
    # Calculate the difference between Newton-Raphson and exact solutions
    difference = np.array(newton_solutions) - np.array(exact_solutions)
    #print(difference)
    
    # Plot the results for comparison
    plot_results(R_t_values, newton_solutions, exact_solutions, difference)


if __name__ == "__main__":
    main(R_0, A, B, C)
 

   