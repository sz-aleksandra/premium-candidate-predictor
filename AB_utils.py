import re
from scipy import stats
import matplotlib.pyplot as plt

def get_logs_results(log_file_path):
	with open(log_file_path, "r") as file:
		log_lines = file.readlines()

	base_pattern = re.compile(r"Base Prediction: (\d+)")
	advanced_pattern = re.compile(r"Advanced Prediction: (\d+)")
	query_pattern = re.compile(r"Zapytanie: (\{.*?\})")

	a_b_experiments = []
	current_experiment = {"Base": [], "Advanced": []}

	for line in log_lines:
		query_match = query_pattern.search(line)
		base_match = base_pattern.search(line)
		advanced_match = advanced_pattern.search(line)
		
		if base_match and current_experiment["Advanced"]:
			a_b_experiments.append(current_experiment)
			current_experiment = {"Base": [], "Advanced": []}
			
		if base_match:
			query = query_match.group(1)
			prediction = int(base_match.group(1))
			current_experiment["Base"].append({"query": query, "prediction": prediction})
		elif advanced_match:
			query = query_match.group(1)
			prediction = int(advanced_match.group(1))
			current_experiment["Advanced"].append({"query": query, "prediction": prediction})

	a_b_experiments.append(current_experiment)

	return a_b_experiments

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