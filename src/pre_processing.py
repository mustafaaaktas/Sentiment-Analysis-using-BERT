import pandas as pd
import re
from nltk.corpus import stopwords


def load_data(file_path):
    """
    Load the dataset from the specified file path.

    Parameters:
        file_path (str): Path to the data file.

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame, or None if loading fails.
    """
    try:
        print(f"Loading data from {file_path}...")
        if file_path.endswith(".zip"):
            data = pd.read_csv(file_path, compression='zip', header=None)
        else:
            data = pd.read_csv(file_path, header=None)
        print("Data loaded successfully!")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def prepare_data(df):
    """
    Prepare the dataset by:
    1. Adding appropriate column headers since raw data doesn't include them.
    2. Dropping unnecessary columns that don't contain useful information.
    3. Renaming remaining columns for clarity.

    Parameters:
        df (pd.DataFrame): Raw dataset to be processed.

    Returns:
        pd.DataFrame: Prepared dataset with only relevant columns, or None if processing fails.
    """
    if df is None:
        print("No data to prepare. Ensure the data is loaded successfully first.")
        return None

    try:
        # Step 1: Define and assign appropriate column names
        df.columns = ["id", "source", "sentiment", "text"]

        # Step 2: Drop unnecessary columns
        df = df.drop(columns=["id", "source"], errors="ignore")

        # Step 3: Rename columns for clarity
        df = df.rename(columns={"sentiment": "sentiment", "text": "text"})

        print("Data prepared successfully!")
        return df
    except Exception as e:
        print(f"Error preparing data: {e}")
        return None


def clean_text(text):
    """
    Clean the input text by:
    1. Removing numbers.
    2. Removing special characters.
    3. Removing stopwords.

    Parameters:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    if isinstance(text, str):
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
        stop_words = set(stopwords.words('english'))  # Load English stopwords
        tokens = text.lower().split()  # Lowercase and tokenize
        text = ' '.join([word for word in tokens if word not in stop_words])  # Remove stopwords
        return text
    else:
        return text


def clean_dataframe_text(df, text_column):
    """
    Apply the clean_text function to a specified text column in the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing text data.
        text_column (str): The name of the column to clean.

    Returns:
        pd.DataFrame: DataFrame with the cleaned text column.
    """
    if text_column in df.columns:
        df[text_column] = df[text_column].apply(clean_text)
        print(f"Text column cleaned successfully!")
    else:
        print(f"Column '{text_column}' not found in DataFrame.")
    return df


def balance_dataset(df, sentiment_column="sentiment", pos_samples=1000, neg_samples=1000, neu_samples=2000):
    """
    Balance the dataset by sampling an equal number of rows for each sentiment class.

    Parameters:
        df (pd.DataFrame): The input DataFrame with text and sentiment columns.
        sentiment_column (str): Name of the sentiment column (default: 'sentiment').
        pos_samples (int): Number of samples to keep for the 'Positive' class.
        neg_samples (int): Number of samples to keep for the 'Negative' class.
        neu_samples (int): Number of samples to keep for the 'Neutral' class.

    Returns:
        pd.DataFrame: Balanced DataFrame with equal samples for each sentiment class.
    """
    try:
        # Filter data by sentiment
        df_positive = df[df[sentiment_column] == "Positive"]
        df_negative = df[df[sentiment_column] == "Negative"]
        df_neutral = df[df[sentiment_column] == "Neutral"]

        # Sample the required number of rows from each class
        df_positive = df_positive.sample(n=pos_samples, random_state=42)
        df_negative = df_negative.sample(n=neg_samples, random_state=42)
        df_neutral = df_neutral.sample(n=neu_samples, random_state=42)

        # Combine the samples into a balanced DataFrame
        balanced_df = pd.concat([df_positive, df_negative, df_neutral]).reset_index(drop=True)
        print("Dataset balanced successfully!")
        return balanced_df
    except Exception as e:
        print(f"Error balancing dataset: {e}")
        return df
