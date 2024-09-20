import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# loss plot
# # Load data from CSV files
# resnet_df = pd.read_csv("resnet18_settings1_10.csv")
# lenet_df = pd.read_csv("lenet_epoch10.csv")
#
# # Ensure the data is converted to NumPy arrays
# resnet_epochs = resnet_df["Epoch"].to_numpy()
# resnet_train_loss = resnet_df["Training Loss"].to_numpy()
# resnet_val_l1_loss = resnet_df["Validation L1 Loss"].to_numpy()
#
# lenet_epochs = lenet_df["Epoch"].to_numpy()
# lenet_train_loss = lenet_df["Training Loss"].to_numpy()
# lenet_val_l1_loss = lenet_df["Validation L1 Loss"].to_numpy()
#
# # ResNet18 Losses Plot
# plt.figure(figsize=(10, 6))
# plt.plot(resnet_epochs, resnet_train_loss, label="Training Loss", linestyle='-', color='blue', marker='o')
# plt.plot(resnet_epochs, resnet_val_l1_loss, label="Validation MSE (L1 Loss)", linestyle=':', color='red', marker='o')
# plt.title("ResNet18 Training Loss vs. Validation MSE (L1 Loss)")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# # plt.grid(True)
# plt.tight_layout()
# plt.savefig("plots/resnet18_losses_plot.png")
#
# # LeNet5 Losses Plot
# plt.figure(figsize=(10, 6))
# plt.plot(lenet_epochs, lenet_train_loss, label="Training Loss", linestyle='-', color='blue', marker='o')
# plt.plot(lenet_epochs, lenet_val_l1_loss, label="Validation MSE (L1 Loss)", linestyle=':', color='red', marker='o')
# plt.title("LeNet5 Training Loss vs. Validation MSE (L1 Loss)")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# # plt.grid(True)
# plt.tight_layout()
# plt.savefig("plots/lenet5_losses_plot.png")



# scatterplot:
# Load the data from the CSV file
data = pd.read_csv("evaluate_resnet18_t1_epoch12.csv", header=None, names=["Observed Sensitivity (dB)", "Estimated Sensitivity (dB)"])

# Scatter plot of observed vs estimated retinal sensitivity
plt.figure(figsize=(8, 6))
plt.scatter(data["Observed Sensitivity (dB)"],
            data["Estimated Sensitivity (dB)"],
            facecolors='none', edgecolors='black',
            linewidths=0.5)
plt.title("Observed vs Estimated Retinal Sensitivity")
plt.xlabel("Observed Sensitivity (dB)")
plt.ylabel("Estimated Sensitivity (dB)")

# Set the axis limits to start at -2 but not show -2
plt.xlim(-2, max(data["Observed Sensitivity (dB)"].max(), data["Estimated Sensitivity (dB)"].max()))
plt.ylim(-2, max(data["Observed Sensitivity (dB)"].max(), data["Estimated Sensitivity (dB)"].max()))

# Adjust tick labels to start after -2
plt.xticks(ticks=plt.xticks()[0][plt.xticks()[0] > -2])  # Remove -2 tick
plt.yticks(ticks=plt.yticks()[0][plt.yticks()[0] > -2])  # Remove -2 tick
#
# # Ensure x and y axes have the same scale
# plt.xlim(min(data["Observed Sensitivity (dB)"].min(), data["Estimated Sensitivity (dB)"].min()),
#          max(data["Observed Sensitivity (dB)"].max(), data["Estimated Sensitivity (dB)"].max()))
# plt.ylim(min(data["Observed Sensitivity (dB)"].min(), data["Estimated Sensitivity (dB)"].min()),
#          max(data["Observed Sensitivity (dB)"].max(), data["Estimated Sensitivity (dB)"].max()))

plt.grid(True)
plt.tight_layout()
# plt.savefig("discuss/plots/scatterplot_observed_vs_estimated.png")
plt.show()


#bland-altman plot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# Load the data from the CSV file
data = pd.read_csv("evaluate_resnet18_t1_epoch12.csv", header=None, names=["Observed Sensitivity (dB)", "Estimated Sensitivity (dB)"])

# Calculate the average and difference between observed and estimated sensitivity
data['Average'] = data[['Observed Sensitivity (dB)', 'Estimated Sensitivity (dB)']].mean(axis=1)
data['Difference'] = data['Observed Sensitivity (dB)'] - data['Estimated Sensitivity (dB)']

# Calculate the mean and standard deviation of the differences
mean_diff = data['Difference'].mean()
std_diff = data['Difference'].std()

# Generate the Bland-Altman plot
plt.figure(figsize=(8, 6))
plt.scatter(data['Average'], data['Difference'], facecolors='none', edgecolors='black', linewidths=0.5)
plt.axhline(mean_diff, color='red', linestyle='--')
plt.axhline(mean_diff + 1.96 * std_diff, color='gray', linestyle='--')
plt.axhline(mean_diff - 1.96 * std_diff, color='gray', linestyle='--')
plt.title("Bland-Altman Plot")
plt.xlabel("Average of Observed and Estimated Sensitivity (dB)")
plt.ylabel("Difference between Observed and Estimated Sensitivity (dB)")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/bland_altman_plot.png")
plt.show()



# Calculate Pearson correlation coefficient
observed = data['Observed Sensitivity (dB)']
predicted = data['Estimated Sensitivity (dB)']

r, _ = pearsonr(observed, predicted)
print(f"Pearson correlation coefficient: {r}")

import pandas as pd
from scipy.stats import wilcoxon

# Load the data from the CSV files
# Assume the CSVs have two columns: 'Observed' and 'Predicted'
model1_df = pd.read_csv('evaluate_resnet18_t1_epoch12.csv')  # Replace with your first CSV file name
model2_df = pd.read_csv('evaluate_lenet_tuned_epoch12.csv')  # Replace with your second CSV file name

# Extract the observed and predicted values
observed1 = model1_df.iloc[:, 0].values  # First column: observed values
predicted1 = model1_df.iloc[:, 1].values  # Second column: predicted values

observed2 = model2_df.iloc[:, 0].values  # First column: observed values (should be the same as observed1)
predicted2 = model2_df.iloc[:, 1].values  # Second column: predicted values

# Check that the observed values are the same in both CSVs
assert (observed1 == observed2).all(), "Observed values must be the same in both CSV files for a paired test."

# Calculate the differences between observed and predicted values for both models
differences_model1 = observed1 - predicted1
differences_model2 = observed2 - predicted2

# Perform the Wilcoxon signed-rank test
stat, p_value = wilcoxon(differences_model1, differences_model2)
print(f"Wilcoxon signed-rank test statistic: {stat}, p-value: {p_value}")
