# 
# Mukund Iyer

# This is a driver script that completes the model building for model
# to predict the popularity of an AirBnB listing. 
# This script will take no input arguments. It two calls, 'clear' to clear
# the directory, and 'all' to run the whole analysis from top to bottom.

# example usage: 
# make all

eda_figures = hist_latitude.png hist_longitude.png hist_availibility_365 hist_calculated_host_listings_count.png minimum_nights.png hist_price.png hist_reviews_per_month.png scatter_long_lat.png scatter_price_min_nights.png scatter_hostlistings_availibility.png correlation.png
eda_paths = $(addprefix results/eda/, $(eda_figures))

evaluations_figs_obs = feat_imp_global.png feat_imp_mag_dir.png test_results.pickle test_results.csv force_plot_1.png force_plot_2.png
eval_paths = $(addprefix results/evaluation/, $(evaluations_figs_obs))

all: $(eval_paths)

# clean data 
data/processed/test.csv data/processed/train.csv: data/raw/AB_NYC_2019.csv src/clean_data.py
	python src/clean_data.py --raw_data_path='./data/raw/AB_NYC_2019.csv' --test_data_path='./data/processed/test.csv' --train_data_path='./data/processed/train.csv'
    
# EDA using cleaned data
# results/eda/hist_latitude.png results/eda/hist_longitude.png results/eda/hist_availibility_365 results/eda/hist_calculated_host_listings_count.png results/eda/minimum_nights.png results/eda/hist_price.png results/eda/hist_reviews_per_month.png results/eda/scatter_long_lat.png results/eda/scatter_price_min_nights.png results/eda/scatter_hostlistings_availibility.png results/eda/correlation.png 
#$(eda_paths): data/processed/train.csv src/eda.py
#	python src/eda.py --train_data_path='./data/processed/train.csv' --results_folder_path='./results'

# preprocessing
results/preprocessing/preprocessor.pickle: data/processed/train.csv data/processed/test.csv src/preprocessing.py
	python src/preprocessing.py --train_data_path='./data/processed/train.csv' --results_folder_path='./results' --test_data_path='./data/processed/test.csv'

# model tuning 
results/model_tuning/tuning_results.pickle results/model_tuning/tuning_results.csv: results/preprocessing/preprocessor.pickle data/processed/X_train.csv data/processed/y_train.csv src/model_tuning.py
	python src/model_tuning.py --processed_data_path='./data/processed' --results_folder_path="./results"

# model evaluation 
$(eval_paths): results/preprocessing/preprocessor.pickle data/processed/X_train.csv data/processed/y_train.csv data/processed/X_test.csv data/processed/y_test.csv src/model_evaluation.py
	python src/model_evaluation.py --processed_data_path='./data/processed' --results_folder_path="./results"



#clean all the intermediate files in data, doc, and results folder.
clean:
	rm -r data/processed
#	rm -f results/eda/*.png
#	rm -f results/preprocessing/*.pickle
#	rm -f results/model_tuning/*.csv
#	rm -f results/model_tuning/*.pickle
#	rm -f results/model_evaluation/*.csv
#	rm -f results/model_evaluation/*.png
#	rm -f results/model_evaluation/*.pickle
	rm -r results/eda
	rm -r results/preprocessing
#	rm -r results/model_tuning
	rm -r results/model_evaluation