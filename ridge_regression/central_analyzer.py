import tenseal as ts
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass

@dataclass
class EncryptedResult:
    """Wrapper for encrypted results to be decrypted by trusted authority"""
    encrypted_XtX: ts.CKKSTensor
    encrypted_Xty: ts.CKKSTensor
    context: ts.Context
    role: str
    feature_names: List[str]

class CentralAnalyzer:
    def __init__(self, public_context: ts.Context):
        self.context = public_context
        self.role_data = {}
    
    def receive_company_data(self, company_name: str, role: str, encrypted_data: Dict[str, ts.CKKSTensor]):
        if encrypted_data is None:
            print(f"WARNING: No data received from {company_name} for role {role}")
            return
            
        if role not in self.role_data:
            self.role_data[role] = []
        self.role_data[role].append(encrypted_data)
        print(f"Received data from {company_name} for role {role}")
    
    def aggregate_encrypted_data_by_role(self, role: str) -> EncryptedResult:
        """Aggregate encrypted data for a specific role"""
        if role not in self.role_data:
            raise ValueError(f"No data available for role: {role}")
        
        if not self.role_data[role]:
            raise ValueError(f"Empty data list for role: {role}")
        
        print(f"\nAggregating data for role: {role}")
        print(f"Number of companies contributing: {len(self.role_data[role])}")
        
        encrypted_XtX_sum = None
        encrypted_Xty_sum = None
        feature_names = self.role_data[role][0]['feature_names']
        scaler_info = self.role_data[role][0]['scaler_info']
        salary_scaler = self.role_data[role][0]['salary_scaler']
        
        for data in self.role_data[role]:
            if encrypted_XtX_sum is None:
                encrypted_XtX_sum = data['encrypted_XtX']
                encrypted_Xty_sum = data['encrypted_Xty']
            else:
                encrypted_XtX_sum += data['encrypted_XtX']
                encrypted_Xty_sum += data['encrypted_Xty']
        
        encrypted_result = EncryptedResult(
            encrypted_XtX=encrypted_XtX_sum,
            encrypted_Xty=encrypted_Xty_sum,
            context=self.context,
            role=role,
            feature_names=feature_names
        )
        # Return scaler_info and salary_scaler along with encrypted_result
        return encrypted_result, scaler_info, salary_scaler
