# 
# Mukund Iyer

# This is a driver script that completes the model building for model
# to predict the popularity of an AirBnB listing. 
# This script will take no input arguments. It two calls, 'clear' to clear
# the directory, and 'all' to run the whole analysis from top to bottom.

# example usage: 
# make all

eda_figures = hist_latitude.png hist_longitude.png hist_availability_365 hist_calculated_host_listings_count.png minimum_nights.png hist_price.png hist_reviews_per_month.png scatter_long_lat.png scatter_price_min_nights.png scatter_hostlistings_availibility.png correlation.png
eda_paths = $(addprefix results/eda/, $(eda_figures))

evaluations_figs_obj = feat_imp_global.png feat_imp_mag_dir.png test_results.pickle test_results.csv force_plot_1.png force_plot_2.png
eval_paths = $(addprefix results/model_evaluation/, $(evaluations_figs_obj))

all: doc/plots.html

# clean data 
data/processed/test.csv data/processed/train.csv: data/raw/AB_NYC_2019.csv src/clean_data.py
	python src/clean_data.py --raw_data_path='./data/raw/AB_NYC_2019.csv' --test_data_path='./data/processed/test.csv' --train_data_path='./data/processed/train.csv'

# preprocessing
results/preprocessing/preprocessor.pickle: data/processed/train.csv data/processed/test.csv src/preprocessing.py
	python src/preprocessing.py --train_data_path='./data/processed/train.csv' --results_folder_path='./results' --test_data_path='./data/processed/test.csv'

# model tuning 
results/model_tuning/tuning_results.pickle results/model_tuning/tuning_results.csv: results/preprocessing/preprocessor.pickle data/processed/X_train.csv data/processed/y_train.csv src/model_tuning.py
	python src/model_tuning.py --processed_data_path='./data/processed' --results_folder_path="./results"

# model evaluation 
$(eval_paths): results/preprocessing/preprocessor.pickle data/processed/X_train.csv data/processed/y_train.csv data/processed/X_test.csv data/processed/y_test.csv src/model_evaluation.py
	python src/model_evaluation.py --processed_data_path='./data/processed' --results_folder_path="./results"

# EDA using cleaned data
$(eda_paths): data/processed/train.csv src/eda.py
	python src/eda.py --train_data_path='./data/processed/train.csv' --results_folder_path='./results'

# model evaluation output
doc/plots.html: $(eda_paths) doc/plots.Rmd results/model_evaluation/feat_imp_global.png results/model_evaluation/feat_imp_mag_dir.png results/model_evaluation/force_plot_1.png results/model_evaluation/force_plot_2.png 
	Rscript -e "rmarkdown::render('doc/plots.Rmd', output_format = 'html_document')" 

#clean all the intermediate files in data, doc, and results folder.
clean:
	rm -r data/processed
	rm -r results/eda
	rm -r results/preprocessing
#	rm -r results/model_tuning
	rm -r results/model_evaluation