import re
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np

def get_logs_results(log_file_path):
	with open(log_file_path, "r") as file:
		log_lines = file.readlines()

	base_pattern = re.compile(r"LogisticRegression")
	advanced_pattern = re.compile(r"RandomForestClassifier")
	prediction_pattern = re.compile(r"Prediction: (\d+)")
	query_pattern = re.compile(r"Zapytanie: (\{.*?\})")
	end_pattern = re.compile(r"GET /predict HTTP/1.1")

	a_b_experiments = []
	current_experiment = {"Base": [], "Advanced": []}

	for line in log_lines:
		query_match = query_pattern.search(line)
		base_match = base_pattern.search(line)
		prediction_match = prediction_pattern.search(line)
		advanced_match = advanced_pattern.search(line)
		end_match = end_pattern.search(line)

		if base_match:
			query = eval(query_match.group(1))
			prediction = int(prediction_match.group(1))
			current_experiment["Base"].append({"query": query, "prediction": prediction})
		elif advanced_match:
			query = eval(query_match.group(1))
			prediction = int(prediction_match.group(1))
			current_experiment["Advanced"].append({"query": query, "prediction": prediction})
		elif end_match:
			a_b_experiments.append(current_experiment)
			current_experiment = {"Base": [], "Advanced": []}

	return a_b_experiments

def find_ground_truth_for_queries(queries, X_test, Y_test):
	ground_truth = []
	for experiment in queries:
		experiment_ground_truth = []
		for query in experiment:
			matching_row = X_test.loc[(X_test == pd.Series(query)).all(axis=1)]

			if not matching_row.empty:
				index = matching_row.index[0]
				experiment_ground_truth.append(int(Y_test.loc[index, 'premium_user']))
		
		ground_truth.append(experiment_ground_truth)
	return ground_truth

def calculate_metrics(base_predictions, base_ground_truth, advanced_predictions, advanced_ground_truth):
	all_metrics = []

	for experiment_no in range(len(base_predictions)):
		base = {
		"ground_truth": np.array(base_ground_truth[experiment_no]),
		"predictions": np.array(base_predictions[experiment_no])
		}

		advanced = {
		"ground_truth": np.array(advanced_ground_truth[experiment_no]),
		"predictions": np.array(advanced_predictions[experiment_no])
		}
		
		# For base model
		base_confusion_matrix = confusion_matrix(base['ground_truth'], base['predictions'])
		base_precision = precision_score(base['ground_truth'], base['predictions'])
		base_accuracy = accuracy_score(base['ground_truth'], base['predictions'])
		base_f1 = f1_score(base['ground_truth'], base['predictions'])
		
		# For advanced model
		advanced_confusion_matrix = confusion_matrix(advanced['ground_truth'], advanced['predictions'])
		advanced_precision = precision_score(advanced['ground_truth'], advanced['predictions'])
		advanced_accuracy = accuracy_score(advanced['ground_truth'], advanced['predictions'])
		advanced_f1 = f1_score(advanced['ground_truth'], advanced['predictions'])
		
		# Collect metrics to dictionary
		all_metrics.append({
		"base": {
		"confusion_matrix": base_confusion_matrix.tolist(),
		"precision": round(base_precision * 100, 3),
		"accuracy": round(base_accuracy * 100, 3),
		"f1_score": round(base_f1 * 100, 3)
		},
		"advanced": {
		"confusion_matrix": advanced_confusion_matrix.tolist(),
		"precision": round(advanced_precision * 100, 3),
		"accuracy": round(advanced_accuracy * 100, 3),
		"f1_score": round(advanced_f1 * 100, 3)
		}
		})

	return all_metrics

def plot_pvalue_changes(base, advanced,plot_title):
	shapiro_p_values = []
	levene_p_values = []
	ttest_p_values = []

	for i in range(3, len(base) + 1):
		base_sample = base[:i]
		advanced_sample = advanced[:i]

		_, shapiro_p_base = stats.shapiro(base_sample)
		_, shapiro_p_adv = stats.shapiro(advanced_sample)
		shapiro_p_values.append((shapiro_p_base, shapiro_p_adv))

		_, levene_p = stats.levene(base_sample, advanced_sample)
		levene_p_values.append(levene_p)

		_, ttest_p = stats.ttest_ind(base_sample, advanced_sample, equal_var=True)
		ttest_p_values.append(ttest_p)

	x = range(3, len(base) + 1)
	shapiro_base_p, shapiro_adv_p = zip(*shapiro_p_values)

	plt.figure(figsize=(12, 8))
	plt.suptitle(plot_title)
	plt.subplot(3, 1, 1)
	plt.plot(x, shapiro_base_p, label='Shapiro-Wilk Base', marker='o')
	plt.plot(x, shapiro_adv_p, label='Shapiro-Wilk Advanced', marker='o')
	plt.axhline(y=0.05, color='r', linestyle='--', label='p = 0.05')
	plt.title('Shapiro-Wilk Test p-values')
	plt.xlabel('Number of Samples')
	plt.ylabel('p-value')
	plt.legend()

	plt.subplot(3, 1, 2)
	plt.plot(x, levene_p_values, label='Levene Test', color='g', marker='o')
	plt.axhline(y=0.05, color='r', linestyle='--', label='p = 0.05')
	plt.title('Levene Test p-values')
	plt.xlabel('Number of Samples')
	plt.ylabel('p-value')
	plt.legend()

	plt.subplot(3, 1, 3)
	plt.plot(x, ttest_p_values, label='t-Test', color='b', marker='o')
	plt.axhline(y=0.05, color='r', linestyle='--', label='p = 0.05')
	plt.title('t-Test p-values')
	plt.xlabel('Number of Samples')
	plt.ylabel('p-value')
	plt.legend()

	plt.tight_layout()
	plt.show()

def create_more_data(first:list, second:list, seed_list, percantage=0.5):
    duplicated_first = []
    duplicated_second = []
    
    for seed in seed_list:
        _,f_sel, _ , s_sel= train_test_split(
            first, second,
            test_size=percantage,
            random_state=seed,
        )
        
        duplicated_first.append(f_sel)
        duplicated_second.append(s_sel)
    
    return duplicated_first,duplicated_second