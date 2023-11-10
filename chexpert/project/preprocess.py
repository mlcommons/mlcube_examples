import pandas as pd
import os
import argparse

class Preprocessor:
	def __init__(self, data_dir):
		self.data_csv_path = os.path.join(data_dir, 'valid.csv')

	def run(self):
		df = pd.read_csv(self.data_csv_path)
		img_path_lists = df['Path'].str.split('/')

		# Ensure the path has not been modified already
		assert len(img_path_lists.iloc[0]) == 5, "Data has already been preprocessed"

		# Modify image path so that it is relative to the file location
		df['Path'] = img_path_lists.str[1:].str.join('/')
		df.to_csv(self.data_csv_path, index=False)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', '--data-dir', type=str, required=True, help='Location of chexpert dataset')
	args = parser.parse_args()
	preprocessor = Preprocessor(args.data_dir)
	preprocessor.run()