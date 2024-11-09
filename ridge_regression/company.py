import pandas as pd
import tenseal as ts
import numpy as np
from typing import Dict, Tuple, List

class Company:
    def __init__(self, name: str, data_path: str, public_context: ts.Context):
        self.name = name
        self.context = public_context
        self.scalers = {}  # Store scalers for each role (Removed as scaling is now centralized)
        self.global_scalers = {}  # To store received global scaling parameters
        try:
            print(f"\nLoading data for {name}...")
            self.raw_data = pd.read_csv(data_path)
            print(f"Data loaded successfully for {name}")
            print(f"Number of records: {len(self.raw_data)}")
            
            if 'Job_Title' not in self.raw_data.columns:
                raise ValueError(f"Job_Title column not found in {name}'s data")
            
            self.raw_data['Job_Title'] = self.raw_data['Job_Title'].str.strip().str.upper()
            print(f"Converting salary data for {name}...")
            self.raw_data['Salary'] = self.raw_data['Salary'].replace(r'[\$,]', '', regex=True).astype(float)
            
            self.roles = self.raw_data['Job_Title'].unique()
            
        except Exception as e:
            print(f"ERROR loading data for {name}: {str(e)}")
            self.raw_data = pd.DataFrame()
            self.roles = []
    
    def compute_data_summary(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compute per role per feature sum, sumsq, count, min, and max for numeric features.
        Returns:
            {role: {feature: {'sum': x, 'sumsq': y, 'count': z, 'min': a, 'max': b}}}
        """
        summary = {}
        numeric_features = ['Age', 'Years_of_Experience', 'Years_of_Tenure', 'Salary']
        for role in self.roles:
            df_role = self.raw_data[self.raw_data['Job_Title'].str.upper() == role.upper()]
            if df_role.empty:
                continue
            role_summary = {}
            for feature in numeric_features:
                feature_values = df_role[feature].astype(float)
                role_summary[feature] = {
                    'sum': feature_values.sum(),
                    'sumsq': (feature_values ** 2).sum(),
                    'count': feature_values.count(),
                    'min': feature_values.min(),
                    'max': feature_values.max()
                }
            summary[role.upper()] = role_summary
        return summary
    
    def set_global_scaling(self, global_scalers: Dict[str, Dict[str, Dict[str, float]]]):
        """
        Receive global scaling parameters.
        
        Args:
            global_scalers: {role: {feature: {'mean': x, 'scale': y}}}
        """
        self.global_scalers = global_scalers
        print(f"{self.name} received global scaling parameters.")
    
    def preprocess_data(self, df: pd.DataFrame, role: str) -> Tuple[np.ndarray, np.ndarray, List[str], Dict, Dict]:
        """
        Preprocess data using global scaling parameters.
        """
        df_role = df[df['Job_Title'].str.upper() == role.upper()].copy()
        
        if len(df_role) == 0:
            raise ValueError(f"No data for role '{role}' in {self.name}")
        
        df_role = df_role.drop(['Employee_ID', 'Job_Title'], axis=1)
        
        # Add derived features
        df_role['experience_ratio'] = df_role['Years_of_Experience'] / (df_role['Age'] - 18)
        df_role['experience_ratio'] = df_role['experience_ratio'].clip(0, 1)

        # Remove 'experience_tenure_ratio' to reduce multicollinearity
        # df_role['experience_tenure_ratio'] = (df_role['Years_of_Experience'] + 1) / (df_role['Years_of_Tenure'] + 1)
        # df_role['experience_tenure_ratio'] = df_role['experience_tenure_ratio'].clip(0.333, 5)
        
        education_categories = ['Bachelor\'s', 'Master\'s', 'Diploma']
        gender_categories = ['Male', 'Female']
        
        for category in education_categories:
            col_name = f'Education_Level_{category}'
            df_role[col_name] = (df_role['Education_Level'] == category).astype(int)
            
        for category in gender_categories:
            col_name = f'Gender_{category}'
            df_role[col_name] = (df_role['Gender'] == category).astype(int)
        
        df_role = df_role.drop(['Gender', 'Education_Level'], axis=1)
        
        # Define numeric features including new derived features
        numeric_features = [
            'Age', 'Years_of_Experience', 'Years_of_Tenure',
            'experience_ratio'
            # Removed 'experience_tenure_ratio' to reduce multicollinearity
        ]
        
        feature_columns = (
            numeric_features + 
            [f'Gender_{g}' for g in gender_categories[:-1]] +
            [f'Education_Level_{e}' for e in education_categories[:-1]]
        )
        
        # Scale numeric features using bounded scaling
        scaler_info = {
            'features': numeric_features,
            'bounds': []
        }
        
        for feature in numeric_features:
            if (role.upper() in self.global_scalers) and (feature in self.global_scalers[role.upper()]):
                bounds = self.global_scalers[role.upper()][feature]
                min_val = bounds['min']
                max_val = bounds['max']
                range_val = max_val - min_val if max_val - min_val != 0 else 1
                
                # Apply scaling to [0, 1] and clip
                df_role[feature] = (df_role[feature] - min_val) / range_val
                df_role[feature] = df_role[feature].clip(0, 1)
                scaler_info['bounds'].append((min_val, max_val))
            else:
                # Default scaling if no bounds available
                df_role[feature] = 0.5
                scaler_info['bounds'].append((0, 1))
        
        X = df_role[feature_columns].values
        y = df_role['Salary'].values
        
        # Scale y using global scaling parameters
        if (role.upper() in self.global_scalers) and ('Salary' in self.global_scalers[role.upper()]):
            salary_bounds = self.global_scalers[role.upper()]['Salary']
            min_salary = salary_bounds['min']
            max_salary = salary_bounds['max']
            salary_range = max_salary - min_salary if max_salary - min_salary != 0 else 1
            y = (y - min_salary) / salary_range
        else:
            # Default scaling if no bounds available
            y = y / y.max()  # Scale to [0,1]
        
        salary_scaler = {
            'min': min_salary,
            'max': max_salary
        }
        
        return X, y, feature_columns, scaler_info, salary_scaler
    
    def encrypt_data_by_role(self, role: str) -> Dict[str, ts.CKKSTensor]:
        """
        Encrypt preprocessed data for a specific role.
        
        Args:
            role: The job role to encrypt data for.
        
        Returns:
            Dictionary containing encrypted XtX, encrypted Xty, feature names, and scaler info.
        """
        try:
            if role.upper() not in [r.upper() for r in self.roles]:
                return None
            
            X, y, feature_names, scaler_info, salary_scaler = self.preprocess_data(self.raw_data, role)
            
            XtX = X.T @ X
            Xty = X.T @ y
            
            XtX_flat = XtX.flatten()
            Xty_flat = Xty.flatten()
            
            encrypted_XtX = ts.ckks_tensor(self.context, XtX_flat)
            encrypted_Xty = ts.ckks_tensor(self.context, Xty_flat)
            
            return {
                'encrypted_XtX': encrypted_XtX,
                'encrypted_Xty': encrypted_Xty,
                'feature_names': feature_names,
                'scaler_info': scaler_info,
                'salary_scaler': salary_scaler  # Include salary scaling info
            }
        except Exception as e:
            print(f"ERROR encrypting data for {self.name}, role {role}: {str(e)}")
            return None
