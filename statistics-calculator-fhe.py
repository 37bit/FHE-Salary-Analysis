import tenseal as ts
import pandas as pd
from typing import Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class EncryptedResult:
    """Wrapper for encrypted results to be decrypted by trusted authority"""
    encrypted_data: Dict[str, Dict[str, ts.CKKSTensor]]
    context: ts.Context

class TrustedAuthority:
    """Handles key generation and final decryption"""
    def __init__(self):
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        self.context.global_scale = 2**40
        self.context.generate_galois_keys()
        self.context.generate_relin_keys()
    
    def get_public_context(self) -> ts.Context:
        """Provide public context to companies and analyzer"""
        return ts.context_from(self.context.serialize(save_secret_key=False))
    
    def decrypt_results(self, encrypted_result: EncryptedResult) -> Dict[str, Dict[str, float]]:
        """Decrypt final results using the secret key"""
        decrypted_benchmarks = {}
        
        for job_title, metrics in encrypted_result.encrypted_data.items():
            decrypted_metrics = {}
            for metric_name, enc_value in metrics.items():
                try:
                    dec_value = enc_value.decrypt(self.context.secret_key())
                    if isinstance(dec_value, ts.PlainTensor):
                        dec_value = dec_value.tolist()[0]
                    decrypted_metrics[metric_name] = float(dec_value)
                except Exception as e:
                    print(f"Error decrypting {metric_name} for {job_title}: {str(e)}")
                    decrypted_metrics[metric_name] = 0.0
            decrypted_benchmarks[job_title] = decrypted_metrics
            
        return decrypted_benchmarks

class Company:
    def __init__(self, name: str, data_path: str, public_context: ts.Context):
        self.name = name
        self.context = public_context
        self.raw_data = pd.read_csv(data_path)
        self.raw_data['Salary'] = self.raw_data['Salary'].str.replace('$', '').str.replace(',', '').astype(float)
        
    def get_raw_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculate raw statistics for validation only"""
        stats = {}
        for job_title in self.raw_data['Job_Title'].unique():
            job_data = self.raw_data[self.raw_data['Job_Title'] == job_title]
            stats[job_title] = {
                'mean_salary': job_data['Salary'].mean(),
                'count': len(job_data),
                'min_salary': job_data['Salary'].min(),
                'max_salary': job_data['Salary'].max()
            }
        return stats

    def encrypt_data(self) -> Dict[str, Dict[str, Tuple[ts.CKKSTensor, float]]]:
        """Encrypt company data"""
        encrypted_data = {}
        
        for job_title in self.raw_data['Job_Title'].unique():
            job_data = self.raw_data[self.raw_data['Job_Title'] == job_title]
            
            # Pre-aggregate data locally before encryption
            salary_sum = job_data['Salary'].sum()
            count = len(job_data)
            
            encrypted_data[job_title] = {
                'salary_sum': ts.ckks_tensor(self.context, [salary_sum]),
                'count': count  # Count isn't sensitive, keep unencrypted
            }

        return encrypted_data

class CentralAnalyzer:
    """Performs computations on encrypted data without access to secret key"""
    def __init__(self, public_context: ts.Context):
        self.context = public_context
        self.company_data = {}

    def receive_company_data(self, company_name: str, encrypted_data: Dict[str, Dict[str, Tuple[ts.CKKSTensor, float]]]):
        self.company_data[company_name] = encrypted_data

    def compute_market_benchmarks(self) -> EncryptedResult:
        """Compute market benchmarks on encrypted data"""
        benchmarks = {}
        
        for job_title in set().union(*(data.keys() for data in self.company_data.values())):
            try:
                total_sum = None
                total_count = 0

                for company_name, company_data in self.company_data.items():
                    if job_title in company_data:
                        company_sum = company_data[job_title]['salary_sum']
                        company_count = company_data[job_title]['count']
                        
                        if total_sum is None:
                            total_sum = company_sum
                        else:
                            total_sum += company_sum
                        
                        total_count += company_count

                if total_count > 0:
                    mean_salary = total_sum * (1.0 / total_count)
                    benchmarks[job_title] = {
                        'mean_salary': mean_salary
                    }

            except Exception as e:
                print(f"Error processing {job_title}: {str(e)}")
                continue

        return EncryptedResult(encrypted_data=benchmarks, context=self.context)

class ResultValidator:
    """Separate validation logic from the FHE implementation"""
    @staticmethod
    def calculate_direct_averages(companies: Dict[str, Company]) -> Dict[str, Dict[str, float]]:
        """Calculate direct averages from raw company data for validation"""
        job_stats = {}
        
        # Collect statistics from all companies
        for company_name, company in companies.items():
            company_stats = company.get_raw_statistics()
            
            for job_title, stats in company_stats.items():
                if job_title not in job_stats:
                    job_stats[job_title] = {
                        'total_salary': 0,
                        'total_count': 0,
                        'min_salary': float('inf'),
                        'max_salary': float('-inf')
                    }
                
                job_stats[job_title]['total_salary'] += stats['mean_salary'] * stats['count']
                job_stats[job_title]['total_count'] += stats['count']
                job_stats[job_title]['min_salary'] = min(job_stats[job_title]['min_salary'], stats['min_salary'])
                job_stats[job_title]['max_salary'] = max(job_stats[job_title]['max_salary'], stats['max_salary'])
        
        # Calculate overall averages
        averages = {}
        for job_title, stats in job_stats.items():
            if stats['total_count'] > 0:
                averages[job_title] = {
                    'mean_salary': stats['total_salary'] / stats['total_count'],
                    'count': stats['total_count'],
                    'min_salary': stats['min_salary'],
                    'max_salary': stats['max_salary']
                }
        
        return averages

def main():
    # Initialize trusted authority
    print("Initializing trusted authority...")
    trusted_authority = TrustedAuthority()
    public_context = trusted_authority.get_public_context()

    # Initialize central analyzer with public context only
    print("Initializing central analyzer...")
    central_analyzer = CentralAnalyzer(public_context)

    # Initialize companies
    print("Initializing companies...")
    companies = {
        'Accenture': Company('Accenture', 'datasets/accenture_employee_salary_data.csv', public_context),
        'GovTech': Company('GovTech', 'datasets/govtech_employee_salary_data.csv', public_context),
        'TikTok': Company('TikTok', 'datasets/tiktok_employee_salary_data.csv', public_context)
    }

    # Calculate direct averages for validation
    print("\nCalculating direct (unencrypted) averages...")
    direct_averages = ResultValidator.calculate_direct_averages(companies)

    # Companies encrypt and send their data
    print("\nCompanies encrypting and sending data...")
    for company_name, company in companies.items():
        try:
            encrypted_data = company.encrypt_data()
            central_analyzer.receive_company_data(company_name, encrypted_data)
            print(f"✓ {company_name} data encrypted and sent successfully")
        except Exception as e:
            print(f"✗ Error with {company_name}: {str(e)}")
            raise e

    # Compute market benchmarks
    print("\nComputing encrypted market benchmarks...")
    try:
        encrypted_benchmarks = central_analyzer.compute_market_benchmarks()
        print("✓ Market benchmarks computed successfully")
    except Exception as e:
        print(f"✗ Error computing benchmarks: {str(e)}")
        return

    # Trusted authority decrypts results
    print("\nTrusted authority decrypting results...")
    decrypted_benchmarks = trusted_authority.decrypt_results(encrypted_benchmarks)

    # Compare results
    print("\nComparing results:")
    for job_title in direct_averages.keys():
        print(f"\nJob Title: {job_title}")
        direct_avg = direct_averages[job_title]['mean_salary']
        encrypted_avg = decrypted_benchmarks[job_title]['mean_salary']
        
        print(f"Direct Average: ${direct_avg:,.2f}")
        print(f"Encrypted Average: ${encrypted_avg:,.2f}")
        print(f"Difference: ${abs(direct_avg - encrypted_avg):,.2f}")
        print(f"Difference Percentage: {(abs(direct_avg - encrypted_avg) / direct_avg) * 100:.2f}%")
        print(f"Sample Size: {direct_averages[job_title]['count']}")
        print(f"Salary Range: ${direct_averages[job_title]['min_salary']:,.2f} - ${direct_averages[job_title]['max_salary']:,.2f}")

if __name__ == "__main__":
    main()