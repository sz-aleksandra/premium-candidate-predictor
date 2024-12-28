import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath, encoding_errors='ignore')
    return data

def train_val_test_split(dataX, dataY, train_ratio=0.75, validation_ratio=0.15, test_ratio=0.1, random_state=42):
    sum_ratios = train_ratio + validation_ratio + test_ratio
    if sum_ratios != 1:
        raise ValueError("ratios do not sum to 1")
    X_train, X_test, Y_train, Y_test = train_test_split(dataX, dataY, test_size=(1 - train_ratio),random_state=random_state)
    val_test_ratio = (test_ratio/(test_ratio + validation_ratio))
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=val_test_ratio,random_state=random_state)
    return ((X_train,Y_train), (X_val,Y_val), (X_test,Y_test))

def save_splits(splits, base_path='content/custom_data'):
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = splits
    
    X_train.to_csv(f'{base_path}/X_train.csv', index=False)
    X_val.to_csv(f'{base_path}/X_val.csv', index=False)
    X_test.to_csv(f'{base_path}/X_test.csv', index=False)
    
    Y_train.to_csv(f'{base_path}/Y_train.csv', index=False)
    Y_val.to_csv(f'{base_path}/Y_val.csv', index=False)
    Y_test.to_csv(f'{base_path}/Y_test.csv', index=False)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")

def main():
    X = load_and_preprocess_data('content/custom_data/processed_X.csv')
    Y = load_and_preprocess_data('content/custom_data/processed_Y.csv')
    
    splits = train_val_test_split(X, Y)
    save_splits(splits)

if __name__ == "__main__":
    main()