import tenseal as ts
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

class TrustedAuthority:
    def __init__(self, lambda_ridge: float = 10.0):  # Further increased regularization
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=16384,
            coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60]
        )
        self.context.global_scale = 2 ** 40
        self.context.generate_galois_keys()
        self.context.generate_relin_keys()
        self.lambda_ridge = lambda_ridge
        self.scalers = {}
        self.global_scalers = {}
    
    def get_public_context(self) -> ts.Context:
        return ts.context_from(self.context.serialize(save_secret_key=False))
    
    def compute_global_scaling(self, company_summaries: Dict):
        """Computing global scaling parameters with improved bounds"""
        print("\n=== Computing Global Scaling Parameters with Improved Bounds ===")
        global_scalers = {}
        
        # Initialize with reasonable bounds
        reasonable_bounds = {
            'Age': (21, 29),
            'Years_of_Experience': (0, 4),
            'Years_of_Tenure': (0, 2),
            'experience_ratio': (0, 1),
            'experience_tenure_ratio': (0.333, 5)
        }
        
        for company_name, roles in company_summaries.items():
            for role, features in roles.items():
                if role not in global_scalers:
                    global_scalers[role] = {}
                for feature, stats in features.items():
                    if feature not in global_scalers[role]:
                        global_scalers[role][feature] = {
                            'sum': 0.0,
                            'sumsq': 0.0,
                            'count': 0,
                            'min': stats['min'],
                            'max': stats['max']
                        }
                    else:
                        global_scalers[role][feature]['sum'] += stats['sum']
                        global_scalers[role][feature]['sumsq'] += stats['sumsq']
                        global_scalers[role][feature]['count'] += stats['count']
                        global_scalers[role][feature]['min'] = min(global_scalers[role][feature]['min'], stats['min'])
                        global_scalers[role][feature]['max'] = max(global_scalers[role][feature]['max'], stats['max'])
        
        # Compute bounded scaling parameters
        for role, features in global_scalers.items():
            for feature, stats in features.items():
                if feature in reasonable_bounds:
                    min_val, max_val = reasonable_bounds[feature]
                else:
                    min_val = stats['min']
                    max_val = stats['max']
                # Use bounded scaling instead of standardization
                global_scalers[role][feature] = {
                    'min': min_val,
                    'max': max_val,
                    'range': max_val - min_val if max_val - min_val != 0 else 1
                }
                
                print(f"Role: {role}, Feature: {feature}, "
                      f"Min: {global_scalers[role][feature]['min']:.4f}, "
                      f"Max: {global_scalers[role][feature]['max']:.4f}")
        
        self.global_scalers = global_scalers
        print("Improved global scaling parameters computed successfully.")

    def distribute_scaling_parameters(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Return the global scaling parameters to distribute to companies.
        
        Returns:
            Global scaling parameters.
        """
        return self.global_scalers
    
    def decrypt_and_train(self, encrypted_result: EncryptedResult) -> Tuple[np.ndarray, List[str]]:
        """Enhanced decrypt and train with improved regularization"""
        XtX_plain = encrypted_result.encrypted_XtX.decrypt(self.context.secret_key())
        Xty_plain = encrypted_result.encrypted_Xty.decrypt(self.context.secret_key())
        
        XtX_plain = np.array(XtX_plain.tolist())
        Xty_plain = np.array(Xty_plain.tolist())
        
        XtX_plain = np.real(XtX_plain)
        Xty_plain = np.real(Xty_plain)
        
        n_features = int(np.sqrt(len(XtX_plain)))
        XtX = XtX_plain.reshape(n_features, n_features)
        Xty = Xty_plain.reshape(n_features, 1)
        
        # Enhanced regularization with feature-specific penalties
        regularization = np.diag([
            self.lambda_ridge if 'Age' in encrypted_result.feature_names[i] or
            'Experience' in encrypted_result.feature_names[i]
            else self.lambda_ridge * 0.5
            for i in range(n_features)
        ])
        
        XtX += regularization
        beta = np.linalg.solve(XtX, Xty)
        
        return beta.flatten(), encrypted_result.feature_names