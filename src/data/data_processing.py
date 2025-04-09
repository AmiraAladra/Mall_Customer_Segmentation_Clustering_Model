import pandas as pd
def load_data(data_path):
    # import the dataset
    try:
        df = pd.read_csv(data_path)
        if df.empty:
            raise ValueError("The provided dataset is empty.")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {data_path}")
    except pd.errors.EmptyDataError:
        raise ValueError("No data found in the file.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the data: {e}")
    



    
    