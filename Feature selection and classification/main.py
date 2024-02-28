import numpy as np
np.seterr(divide='ignore',invalid='ignore')
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, f1_score
from losses.jFitnessFunction import jFitnessFunction
from Grid_search_Classifer import *
from feature_selection.jChameleonSwarmAlgorithm import jChameleonSwarmAlgorithm

###########################################################
# Swarm intelligence algorithm
from feature_selection.jChameleonSwarmAlgorithm import jChameleonSwarmAlgorithm
from feature_selection.jDungBeetleOptimizer import jDungBeetleOptimizer
from feature_selection.jSandCatSwarmOptimization import jSandCatSwarmOptimization
from feature_selection.jDynamicArithmeticOptimizationAlgorithm import jDynamicArithmeticOptimizationAlgorithm
from feature_selection.jBelugaWhaleOptimization import jBelugaWhaleOptimization
from feature_selection.jBinaryDwarfMongooseOptimizer import jBinaryDwarfMongooseOptimizer
###########################################################

###########################################################
plt.close('all')
# Number of k in K-nearest neighbor
opts = {'k': 20}
# Ratio of validation data
ho = 0.2
# Common parameter settings
opts['N'] = 5     # number of solutions
opts['T'] = 2    # maximum number of iterations
opts['B'] = 0.1 # Constriction Coefeicient
# Parameter of WOA
opts['b'] = 1
# DBO algorithm parameters
opts['P_percent'] = 1
###########################################################

# Datasets list
sw=['ST', 'GK', 'MT', 'CG', 'BS', 'EC']

# Feature coding
data_t=['BINA_3D']

# Feature selection algorithm
select_features=['CSAO','BDMO','BWO','SCSO','DBO','DAOA']

for p in range(len(select_features)):

	for k in range(len(data_t)):

		# Set data path
		data_path = f'datasets\\{data_t[k]}\\'

		for i in range(len(sw)):

			# Read data (positive and negative samples, training set, and test set)
			Negative_ind = pd.read_csv(data_path + 'val\\Negative\\' + sw[i] + '_Negative_val.csv').values
			Negative_train = pd.read_csv(data_path + 'train\\Negative\\' + sw[i] + '_Negative_train.csv').values
			Positive_ind = pd.read_csv(data_path  + 'val\\Positive\\'+ sw[i] + '_Positive_val.csv').values
			Positive_train = pd.read_csv(data_path  + 'train\\Positive\\'+ sw[i] + '_Positive_train.csv').values

			# Combine data
			x_test = np.concatenate((Negative_ind, Positive_ind), axis=0)
			y_test = np.concatenate((np.zeros((Negative_ind.shape[0], 1)), np.ones((Positive_ind.shape[0], 1))), axis=0)
			y_test = np.squeeze(y_test)  # Reshape the label dimension from (n, 1) to (n,)
			x_train = np.concatenate((Negative_train, Positive_train), axis=0)
			y_train = np.concatenate((np.zeros((Negative_train.shape[0], 1)), np.ones((Positive_train.shape[0], 1))), axis=0)
			y_train = np.squeeze(y_train)  # Reshape the label dimension from (n, 1) to (n,)
				
			# Select XGBoost as the best-performing model (after grid search and cross-validation)
			model_xgb, best_params_xgb = train_xgb_with_grid_search(x_train, y_train, param_grid_xgb)
			model = model_xgb
			print(f'The best parameters on the {sw[i]} dataset are: {best_params_xgb}')

			# Create a fitness function object.
			fitness = jFitnessFunction(model, num_feat=x_train.shape[1], alpha=0.9, beta=0.1)

			# Perform feature selection using 6 swarm intelligence algorithms (after grid search)
			if select_features[p]=='DBO':
				FSModel = jDungBeetleOptimizer(N=60, max_Iter=30, loss_func=fitness, thres=0.5, tau=1, rho=0.2, eta=1)
			elif select_features[p]=='SCSO':
				FSModel = jSandCatSwarmOptimization(N=60, max_Iter=30, loss_func=fitness, thres=0.5, tau=1, rho=0.2, eta=1)
			elif select_features[p]=='DAOA':
				FSModel = jDynamicArithmeticOptimizationAlgorithm(N=60, max_Iter=60, loss_func=fitness, thres=0.5, tau=1, rho=0.2, eta=1)
			elif select_features[p]=='BWO':
				FSModel = jBelugaWhaleOptimization(N=60, max_Iter=30, loss_func=fitness, thres=0.5, tau=1, rho=0.2, eta=1)
			elif select_features[p]=='BDMO':
				FSModel = jBinaryDwarfMongooseOptimizer(N=60, max_Iter=30, loss_func=fitness, thres=0.5, tau=1, rho=0.2, eta=1)
			elif select_features[p]=='CSAO':
				FSModel = jChameleonSwarmAlgorithm(N=60, max_Iter=30, loss_func=fitness, thres=0.5, tau=1, rho=0.2, eta=1)

			history = FSModel.optimize(x_train, x_test, y_train, y_test)
			sf_idx = history['sf']

			# Plot the line chart of loss change
			plt.plot(history['c'], label=f'{sw[i]}_{select_features[p]}')

			# Set legend and title
			plt.legend()
			plt.title(f'Loss Change Over Epochs - Algorithm: {select_features[p]} - Dataset: {sw[i]}')
			plt.xlabel('Epoch')
			plt.ylabel('Loss')

			# Create the path to save the loss change plot
			pic_path = f'log//{select_features[p]}//{data_t[k]}//'
			os.makedirs(pic_path, exist_ok=True)  # Create the directory if it does not exist

			# Save the figure
			plt_path = os.path.join(pic_path, f'loss_plot_{select_features[p]}_{sw[i]}.png')
			plt.savefig(plt_path)
			plt.close()

			# Ensure that indices in sf_idx do not exceed the number of columns in x_train
			sf_idx = [idx for idx in sf_idx if idx < x_train.shape[1]]

			if len(sf_idx) == 0:
				print(f'Under the XGB model, the number of selected features for {select_features[p]} algorithm is 0')
			else:
				model.fit(x_train[:, sf_idx], y_train)
				
				# Construct the save path
				log_path = f'log//{select_features[p]}//{data_t[k]}//'
				os.makedirs(log_path, exist_ok=True)  # Create the directory if it does not exist

				# Generate and save the text file
				with open(os.path.join(log_path, f'{sw[i]}.txt'), 'a') as f:
					for j in range(2):
						X = x_train if j == 0 else x_test
						Y = y_train if j == 0 else y_test
						if j == 0:
							y_pred = model.predict(X[:, sf_idx])
							f.write(f'Selected feature indices on the {sw[i]} dataset: {sf_idx}\n\n')
						elif j == 1:
							y_pred = model.predict(X[:, sf_idx])
						
						# Calculate the confusion matrix
						cm = confusion_matrix(Y, y_pred)
						# Calculate MCC (Matthews Correlation Coefficient)
						tn, fp, fn, tp = cm.ravel()
						mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
						# Calculate accuracy
						acc = accuracy_score(Y, y_pred)
						# Calculate Sn (Sensitivity)
						sn = tp / (tp + fn)
						# Calculate Sp (Specificity)
						sp = tn / (tn + fp)
						# Calculate Precision
						precision = precision_score(Y, y_pred)
						# Calculate F1 score
						f1 = f1_score(Y, y_pred)
						# Calculate AUC (Area Under the ROC Curve)
						auc = roc_auc_score(Y, y_pred)
						if j == 0:
							f.write(f'The best parameters on the {sw[i]} dataset are: {best_params_xgb}\n\n')

						if j == 0:
							f.write("Metrics on the training set:\n")
						else:
							f.write("Metrics on the test set:\n")

						f.write(f"MCC: {mcc}\n")
						f.write(f"Accuracy: {acc}\n")
						f.write(f"Sensitivity: {sn}\n")
						f.write(f"Specificity: {sp}\n")
						f.write(f"Precision: {precision}\n")
						f.write(f"F1 Score: {f1}\n")
						f.write(f"AUC: {auc}\n\n")
					print(f'Under the conditions of the feature selection algorithm {select_features[p]}: XGB algorithm has completed on the {sw[i]} dataset')