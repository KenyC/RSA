import matplotlib.pyplot as plt
import numpy as np

from exh.utils.table import Table


def heatmap(np_array, lab_lines, lab_cols, round_to = 3):
	rounded_array = np.round(np_array, round_to)

	fig, ax = plt.subplots()
	im = ax.imshow(rounded_array)

	# We want to show all ticks...
	ax.set_xticks(np.arange(len(lab_cols)))
	ax.set_yticks(np.arange(len(lab_lines)))
	# ... and label them with the respective list entries
	ax.set_xticklabels(lab_cols)
	ax.set_yticklabels(lab_lines)

	ax.tick_params(top=True, bottom=False,
	               labeltop=True, labelbottom=False)


	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=-45, ha="right",
	         rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	for i in range(len(lab_lines)):
	    for j in range(len(lab_cols)):
	        text = ax.text(j, i, rounded_array[i, j],
	                       ha="center", va="center", color="w")

	fig.tight_layout()
	plt.show()

def table(np_array, lab_lines, lab_cols, round_to = 3):
	table = Table(char_bold_col = "||")
	table.set_header([""] + list(lab_cols))
	table.set_strong_col(1)
	rounded_array = np.round(np_array, round_to)

	for line, label in zip(rounded_array, lab_lines):
		table.add_row([label] + [str(value) for value in line]) 

	table.print_console()
