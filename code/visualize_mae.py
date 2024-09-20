import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t

# Load the CSV files
resnet_df = pd.read_csv('evaluate_resnet18_t1_epoch12.csv')
lenet_df = pd.read_csv('evaluate_lenet_tuned_epoch12.csv')

# Calculate Mean Absolute Error (MAE) for each model
resnet_mae = np.abs(resnet_df.iloc[:, 0] - resnet_df.iloc[:, 1])
lenet_mae = np.abs(lenet_df.iloc[:, 0] - lenet_df.iloc[:, 1])

# Calculate the mean and 95% confidence interval for MAE
resnet_mean = np.mean(resnet_mae)
lenet_mean = np.mean(lenet_mae)

# Standard error of the mean (SEM)
resnet_sem = sem(resnet_mae)
lenet_sem = sem(lenet_mae)

# 95% confidence interval
confidence_level = 0.95
degrees_freedom_resnet = len(resnet_mae) - 1
degrees_freedom_lenet = len(lenet_mae) - 1

# Calculate the t critical value
t_critical_resnet = t.ppf((1 + confidence_level) / 2, degrees_freedom_resnet)
t_critical_lenet = t.ppf((1 + confidence_level) / 2, degrees_freedom_lenet)

# Margin of error
resnet_ci = t_critical_resnet * resnet_sem
lenet_ci = t_critical_lenet * lenet_sem

# Plot the mean absolute error with 95% CI (only error bars in one vertical line)
x_position = 1  # Single x position for both error bars
means = [resnet_mean, lenet_mean]
ci_values = [resnet_ci, lenet_ci]
labels = ['newResNet18', 'LeNet5']

plt.figure(figsize=(4, 6))

# Plot each error bar at the same x position
for i, (mean, ci, label) in enumerate(zip(means, ci_values, labels)):
    plt.errorbar(x_position, mean, yerr=ci, fmt='o', capsize=10, color='black', ecolor='black', elinewidth=2, capthick=2)
    # Annotate the label next to each error bar
    plt.text(x_position + 0.02, mean, label, va='center', fontsize=12)

# Set y-axis properties
plt.ylabel('Mean Absolute Error')
plt.title('Mean Absolute Error with 95% Confidence Interval')
plt.ylim(4, 6)
plt.xticks([])  # Hide x-axis ticks

plt.show()

# print(resnet_mean-resnet_ci,resnet_mean+resnet_ci)
print(lenet_mean-lenet_ci,lenet_mean+lenet_ci)
print(lenet_mean)
