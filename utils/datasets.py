from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class Datasets:
    def __init__(self):
        self.path = {
            'mnist': {
                'train': 'data/mnist/mnist_train.csv',
                'test': 'data/mnist/mnist_test.csv'
            },
            'fashion_mnist': {
                'train': 'data/fashion-mnist/fashion-mnist_train.csv',
                'test': 'data/fashion-mnist/fashion-mnist_test.csv'
            },
            'boston_housing': {
                'dataset': 'data/boston_housing/BostonHousing.csv'
            },
            'iris': {
                'dataset': 'data/iris/Iris.csv'
            }
        }
    
    def __call__(self, dataset_name: str):
        return self.get_dataset(dataset_name)
            
    def get_dataset(self, dataset_name: str) -> tuple:
        if dataset_name == 'mnist':
            train_data = pd.read_csv(self.path['mnist']['train'], low_memory=False)
            X = train_data.drop('label', axis=1).values.astype('float32') / 255.0
            y = train_data['label'].values.astype(int)
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=0.25,
                random_state=42, 
                shuffle=True,
                stratify=y
            )
            
            test_data = pd.read_csv(self.path['mnist']['test'], low_memory=False)
            X_test = test_data.drop('label', axis=1).values.astype('float32') / 255.0
            y_test = test_data['label'].values.astype(int)

        elif dataset_name == 'fashion_mnist':
            train_data = pd.read_csv(self.path['fashion_mnist']['train'], low_memory=False)
            X = train_data.drop('label', axis=1).values / 255.0
            y = train_data['label'].values.astype(int)
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=0.25,
                random_state=42, 
                shuffle=True,
                stratify=y
            )
            test_data = pd.read_csv(self.path['fashion_mnist']['test'], low_memory=False)
            X_test = test_data.drop('label', axis=1).values / 255.0
            y_test = test_data['label'].values.astype(int)

        elif dataset_name == 'boston_housing':
            data = pd.read_csv(self.path['boston_housing']['dataset'])
            
            if data.isnull().any().any():
                data = data.fillna(data.mean())
            
            X = data.drop('medv', axis=1).values.astype('float32')
            y = data['medv'].values.astype('float32')
            
            if np.any(np.isnan(X)):
                print("NaN found in X after conversion, replacing with column means...")
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='mean')
                X = imputer.fit_transform(X)
            
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y,
                test_size=0.4,
                random_state=42,
                shuffle=True
            )
            
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size=0.5,
                random_state=42,
                shuffle=True
            )
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
            
            y_train = y_train.reshape(-1, 1)
            y_val = y_val.reshape(-1, 1) 
            y_test = y_test.reshape(-1, 1)
        
        else:
            raise ValueError(f"Dataset '{dataset_name}' não reconhecido. Opções: 'mnist', 'fashion_mnist', 'boston_housing', 'iris'")
        
        return X_train, X_val, X_test, y_train, y_val, y_test