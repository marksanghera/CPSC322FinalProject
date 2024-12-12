##############################################
# Programmer: Leo Jia
# Class: CPSC 322-01, Fall 2024
# Programming Assignment #3
# 10/8/24
# 
# Description: This program does programmy stuff
# and plot utils and stuff like that
##############################################

import matplotlib.pyplot as plt

def calculate_frequencies(column_data):
    unique_values = []
    frequencies = []
    for value in column_data:
        if value not in unique_values:
            unique_values.append(value)
            frequencies.append(1)
        else:
            index = unique_values.index(value)
            frequencies[index] += 1
    return unique_values, frequencies

def plot_discretized_frequency_diagram(attribute, column_data, label):
    unique_values, frequencies = calculate_frequencies(column_data)
    plt.bar(unique_values, frequencies)
    plt.title(f"Total Number by {attribute}")
    plt.xlabel(attribute)
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.text(-0.1, -0.1, label, fontsize=12, ha='center', transform=plt.gca().transAxes)
    plt.show()

def discretize_mpg_doe(mpg_values):
    categories = []
    for mpg in mpg_values:
        if mpg >= 45:
            categories.append(10)
        elif mpg >= 37:
            categories.append(9)
        elif mpg >= 31:
            categories.append(8)
        elif mpg >= 27:
            categories.append(7)
        elif mpg >= 24:
            categories.append(6)
        elif mpg >= 20:
            categories.append(5)
        elif mpg >= 17:
            categories.append(4)
        elif mpg >= 15:
            categories.append(3)
        elif mpg == 14:
            categories.append(2)
        else:
            categories.append(1)
    return categories

def discretize_mpg_equal_width(mpg_values, num_bins=5):
    min_mpg = min(mpg_values)
    max_mpg = max(mpg_values)
    bin_width = (max_mpg - min_mpg) / num_bins
    categories = []
    for mpg in mpg_values:
        bin_number = int((mpg - min_mpg) // bin_width) + 1
        if bin_number > num_bins:
            bin_number = num_bins
        categories.append(bin_number)
    return categories

def plot_frequency_diagram(attribute, categories, labels=None):
    unique_values = sorted(set(categories))
    frequencies = [categories.count(value) for value in unique_values]

    if labels:
        labels = [labels[unique_values.index(value)] for value in unique_values]

    plt.bar(unique_values, frequencies, tick_label=labels if labels else unique_values, color='steelblue')
    plt.title(f"Total Number by {attribute.capitalize()}")
    plt.xlabel(attribute.capitalize())
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_histogram(attribute, column_data, label, bins=10):
    column_data = [float(value) for value in column_data if value != 'NA']
    
    plt.hist(column_data, bins=bins, edgecolor = 'black')
    plt.title(f"Distribution of {attribute} Values")
    plt.xlabel(attribute)
    plt.ylabel("Count")
    plt.text(-0.1, -0.1, label, fontsize=12, ha='center', transform=plt.gca().transAxes)
    plt.show()


def calculate_regression(x, y):
    """Calculate the slope, intercept, and correlation coefficient manually without using math."""
    n = len(x)
    
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2 = sum(xi ** 2 for xi in x)
    sum_y2 = sum(yi ** 2 for yi in y)
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    intercept = (sum_y - slope * sum_x) / n
    
    r_numerator = (n * sum_xy - sum_x * sum_y)
    r_denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5
    correlation_coef = r_numerator / r_denominator if r_denominator != 0 else 0
    
    return slope, intercept, correlation_coef

def filter_valid_pairs(x, y):
    """Filter out invalid pairs (i.e., where either x or y is missing)."""
    valid_x = []
    valid_y = []
    for xi, yi in zip(x, y):
        if xi != 'NA' and yi != 'NA':
            valid_x.append(xi)
            valid_y.append(yi)
    return valid_x, valid_y

def plot_scatter_with_regression(x, y, x_label, y_label, label):
    """Plot a scatter plot with a manually calculated regression line and correlation coefficient."""
    x, y = filter_valid_pairs(x, y)
    
    plt.scatter(x, y, color='blue')
    
    slope, intercept, correlation_coef = calculate_regression(x, y)
    
    regression_line = [slope * xi + intercept for xi in x]
    
    plt.plot(x, regression_line, color='red')
    
    plt.title(f"{x_label.capitalize()} vs {y_label.lower()}")
    plt.xlabel(x_label.capitalize())
    plt.ylabel(y_label.capitalize())
    plt.text(-0.1, -0.1, label, fontsize=12, ha='center', transform=plt.gca().transAxes)
    plt.annotate(f"r={correlation_coef:.2f}", xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=12, color='red', bbox=dict(boxstyle="round,pad=0.3", edgecolor='red', facecolor='white'))

    plt.grid(True)
    plt.tight_layout()
    plt.show()