from src.pre_processing import load_data, prepare_data, clean_dataframe_text, balance_dataset

# Load and prepare the training data
TRAIN_DATA_PATH = ".venv/twitter_training.csv.zip"
train_data = load_data(TRAIN_DATA_PATH)
train_data_prepared = prepare_data(train_data)

# Clean the text column
if train_data_prepared is not None:
    train_data_cleaned = clean_dataframe_text(train_data_prepared, "text")

    # Balance the dataset using the correct 'sentiment' column
    train_data_balanced = (
        balance_dataset(train_data_cleaned,
                        sentiment_column="sentiment",
                        pos_samples=1000,
                        neg_samples=1000,
                        neu_samples=2000))
    print("\nBalanced training data:")
    print(train_data_balanced['sentiment'].value_counts())
else:
    print("Failed to prepare the training data.")
