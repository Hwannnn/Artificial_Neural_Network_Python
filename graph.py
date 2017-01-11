import csv
import numpy as np
from scipy.interpolate import spline
import matplotlib.pyplot as plt

filename = "_wdbc_cost_function.csv"

cost_function = []
for row in csv.reader(open(filename)):
	cost_function.append(row)
del cost_function[0]
for row in range(len(cost_function)):
	del cost_function[row][0]
	cost_function[row][:] = [float(x) for x in cost_function[row][:]]

epoch = []
train = []
validation = []
for row in cost_function:
	epoch.append(row[0])
	train.append(row[1])
	validation.append(row[2])

epoch = np.array(epoch)
train = np.array(train)
validation = np.array(validation)

epoch_smooth = np.linspace(epoch.min(), epoch.max(), num=10000)
train_smooth = spline(epoch, train, epoch_smooth)
validation_smooth = spline(epoch, validation, epoch_smooth)

plt.plot(epoch_smooth, train_smooth, label="Train")
plt.plot(epoch_smooth, validation_smooth, label="Validation")

plt.xlabel("Epoch")
plt.ylabel("Error rate")
plt.title("Node=20 learning_rate=0.0001 momentum=0.8")
plt.legend(loc="upper right")

plt.grid(True)

plt.savefig("_wdbc_ann.png")