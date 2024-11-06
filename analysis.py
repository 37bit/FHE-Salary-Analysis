import tenseal as ts
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class EncryptedAnalysis:
    """Privacy-protected cross-company analysis results"""
    # Market position analysis
    encrypted_percentiles: ts.CKKSTensor      # Role-specific salary percentiles
    encrypted_experience_curves: ts.CKKSTensor # Experience-compensation curves
    encrypted_pay_gaps: ts.CKKSTensor         # Protected group analysis
    # Number of data points in each group (for privacy thresholds)
    group_counts: Dict[str, int]
    context: ts.Context

class MarketAnalysis:
    """Secure market analysis with privacy protections"""
    
    # Minimum threshold of data points needed to report a statistic
    MIN_GROUP_SIZE = 5
    # Noise scale for differential privacy
    NOISE_SCALE = 0.1
    
    def __init__(self, 
                company_name: str,
                company_data: pd.DataFrame,  # Add this parameter
                analysis_results: EncryptedAnalysis,
                feature_names: List[str]):
        self.company_name = company_name
        self.company_data = company_data  # Store company data
        self.results = analysis_results
        self.feature_names = feature_names
        
    def get_market_insights(self) -> Dict[str, List[str]]:
        """Generate privacy-preserving market insights"""
        insights = {
            'market_position': [],
            'compensation_drivers': [],
            'equity_analysis': []
        }
        
        # Only analyze groups with sufficient data
        role_insights = self._analyze_market_position()
        if role_insights:
            insights['market_position'].extend(role_insights)
            
        driver_insights = self._analyze_compensation_drivers()
        if driver_insights:
            insights['compensation_drivers'].extend(driver_insights)
            
        equity_insights = self._analyze_pay_equity()
        if equity_insights:
            insights['equity_analysis'].extend(equity_insights)
            
        return insights
        
    def _analyze_market_position(self) -> List[str]:
        """Analyze company's position relative to market"""
        insights = []
        
        # Get decrypted market-wide medians
        market_medians = np.array(self.results.encrypted_percentiles.decrypt().tolist())
        
        roles = ['Software Engineer', 'Data Scientist', 'Product Manager']
        for i, role in enumerate(roles):
            if self.results.group_counts.get(role, 0) >= self.MIN_GROUP_SIZE:
                # Calculate company's median for this role
                role_data = self.company_data[self.company_data['Job_Title'] == role]
                if not role_data.empty:
                    company_median = role_data['Salary'].median()
                    relative_position = (company_median / market_medians[i]) - 1
                    
                    if abs(relative_position) < 0.1:
                        market_position = "at market rate"
                    elif relative_position > 0:
                        market_position = f"above market rate (+{relative_position:.1%})"
                    else:
                        market_position = f"below market rate ({relative_position:.1%})"
                        
                    insights.append(
                        f"{role} median compensation is {market_position}"
                    )
        
        return insights
    
    def _analyze_compensation_drivers(self) -> List[str]:
        """Analyze what drives compensation differences"""
        insights = []
        
        # Get experience curves and add noise
        exp_curves = np.array(self.results.encrypted_experience_curves.decrypt().tolist())
        exp_curves += np.random.normal(0, self.NOISE_SCALE, exp_curves.shape)
        
        # Reshape to (n_roles, n_experience_levels)
        exp_curves = exp_curves.reshape(3, 5)  # 3 roles, 5 experience levels
        
        roles = ['Software Engineer', 'Data Scientist', 'Product Manager']
        for i, role in enumerate(roles):
            if self.results.group_counts.get(role, 0) >= self.MIN_GROUP_SIZE:
                # Calculate average yearly increase
                yearly_increases = np.diff(exp_curves[i])
                avg_increase = np.mean(yearly_increases[yearly_increases != 0])
                
                insights.append(
                    f"{role} experience premium: {avg_increase:.1%} per year "
                    f"({'above' if avg_increase > 0.08 else 'below'} industry average of 8%)"
                )
                
        return insights
    
    def _analyze_pay_equity(self) -> List[str]:
        """Analyze pay equity across groups"""
        insights = []
        
        # Get pay gap analysis and add noise
        pay_gaps = np.array(self.results.encrypted_pay_gaps.decrypt().tolist())
        pay_gaps += np.random.normal(0, self.NOISE_SCALE, pay_gaps.shape)
        
        # Only report if we have enough data points
        if all(count >= self.MIN_GROUP_SIZE for count in self.results.group_counts.values()):
            gender_gap = pay_gaps[0]
            edu_gap = np.mean(pay_gaps[1:])  # Average education premium
            
            if abs(gender_gap) > 0.05:  # 5% threshold
                insights.append(
                    f"Compensation gap: {abs(gender_gap):.1%} "
                    f"({'above' if abs(gender_gap) > 0.032 else 'at'} industry average of 3.2%)"
                )
            
            insights.append(
                f"Education premium: {edu_gap:.1%} "
                f"({'above' if edu_gap > 0.15 else 'below'} industry average of 15%)"
            )
            
        return insights

class Company:
    def __init__(self, name: str, data_path: str, public_context: ts.Context):
        self.name = name
        self.context = public_context
        self.raw_data = pd.read_csv(data_path)
        self.raw_data['Salary'] = self.raw_data['Salary'].str.replace(r'[\$,]', '', regex=True).astype(float)
        
        # Preprocess data
        self.X, self.y, self.feature_names = self._preprocess_data()
        
    def _preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Preprocess data focusing on meaningful comparisons"""
        df = self.raw_data.copy()
        
        # Calculate role-specific salary percentiles
        df['SalaryPercentile'] = df.groupby('Job_Title')['Salary'].transform(
            lambda x: pd.qcut(x, q=100, labels=False, duplicates='drop') / 100
        )
        
        # Create experience buckets with custom bins to avoid duplicates
        experience_bins = [0, 1, 2, 3, 4, float('inf')]
        experience_labels = ['0-1', '1-2', '2-3', '3-4', '4+']
        df['ExperienceBucket'] = pd.cut(
            df['Years_of_Experience'],
            bins=experience_bins,
            labels=experience_labels,
            include_lowest=True
        )
        
        # Standardize education levels
        education_map = {
            "Bachelor's": 'Bachelors',
            'Diploma': 'Diploma',
            "Master's": 'Masters'
        }
        df['Education_Level'] = df['Education_Level'].map(education_map)
        
        # Extract features
        features = ['Job_Title', 'Gender', 'Education_Level', 
                'Years_of_Experience', 'Years_of_Tenure']
        
        # One-hot encode categoricals
        df_encoded = pd.get_dummies(df[features])
        
        return (df_encoded.values, 
                df['SalaryPercentile'].values, 
                df_encoded.columns.tolist())
    
    def encrypt_analysis(self) -> Dict[str, ts.CKKSTensor]:
        """Encrypt metrics for secure analysis"""
        # Calculate metrics for each role
        role_metrics = self._calculate_role_metrics()
        experience_curves = self._calculate_experience_curves()
        equity_metrics = self._calculate_equity_metrics()
        
        # Count data points in each group for privacy thresholds
        group_counts = self.raw_data['Job_Title'].value_counts().to_dict()
        
        # Encrypt everything except counts
        encrypted_data = {
            'encrypted_percentiles': ts.ckks_tensor(self.context, role_metrics),
            'encrypted_exp_curves': ts.ckks_tensor(self.context, experience_curves),
            'encrypted_equity': ts.ckks_tensor(self.context, equity_metrics),
            'group_counts': group_counts
        }
        
        return encrypted_data
    
    def _calculate_role_metrics(self) -> np.ndarray:
        """Calculate normalized role-specific salary metrics"""
        metrics = []
        
        for role in ['Software Engineer', 'Data Scientist', 'Product Manager']:
            role_data = self.raw_data[self.raw_data['Job_Title'] == role]
            if not role_data.empty:
                metrics.append(role_data['Salary'].median())  # Just send raw medians
            else:
                metrics.append(0)
        return np.array(metrics)
    
    def _calculate_experience_curves(self) -> np.ndarray:
        """Calculate normalized experience-to-compensation curves"""
        curves = []
        
        for role in ['Software Engineer', 'Data Scientist', 'Product Manager']:
            role_data = self.raw_data[self.raw_data['Job_Title'] == role]
            if not role_data.empty:
                entry_level_salary = role_data[role_data['Years_of_Experience'] <= 1]['Salary'].median()
                if pd.isna(entry_level_salary) or entry_level_salary == 0:
                    entry_level_salary = role_data['Salary'].min()
                
                exp_salary = []
                for exp in range(5):  # 5 experience levels
                    salary = role_data[
                        (role_data['Years_of_Experience'] >= exp) & 
                        (role_data['Years_of_Experience'] < exp + 1)
                    ]['Salary'].median()
                    
                    if pd.isna(salary):
                        exp_salary.append(0)
                    else:
                        # Calculate as percentage increase from entry level
                        exp_salary.append((salary / entry_level_salary - 1) if entry_level_salary else 0)
                curves.extend(exp_salary)
            else:
                curves.extend([0] * 5)
                
        return np.array(curves)
    
    def _calculate_equity_metrics(self) -> np.ndarray:
        """Calculate normalized equity-related metrics"""
        metrics = []
        
        # Gender pay gap (as percentage difference)
        male_median = self.raw_data[self.raw_data['Gender'] == 'Male']['Salary'].median()
        female_median = self.raw_data[self.raw_data['Gender'] == 'Female']['Salary'].median()
        if male_median and female_median:
            metrics.append((male_median - female_median) / male_median)
        else:
            metrics.append(0)
        
        # Education premium (as percentage increase)
        base_salary = self.raw_data[self.raw_data['Education_Level'] == 'Diploma']['Salary'].median()
        bachelors_salary = self.raw_data[self.raw_data['Education_Level'] == "Bachelor's"]['Salary'].median()
        masters_salary = self.raw_data[self.raw_data['Education_Level'] == "Master's"]['Salary'].median()
        
        if base_salary and bachelors_salary:
            metrics.append((bachelors_salary - base_salary) / base_salary)
        else:
            metrics.append(0)
            
        if bachelors_salary and masters_salary:
            metrics.append((masters_salary - bachelors_salary) / bachelors_salary)
        else:
            metrics.append(0)
                
        return np.array(metrics)

class IndustryAnalyzer:
    """Privacy-preserving industry-wide analysis"""
    def __init__(self, public_context: ts.Context):
        self.context = public_context
        self.company_data = []
        self.company_names = []
    
    def add_company_data(self, company_name: str, encrypted_data: Dict[str, ts.CKKSTensor]):
        self.company_data.append(encrypted_data)
        self.company_names.append(company_name)
    
    def analyze(self) -> EncryptedAnalysis:
        """Perform privacy-preserving industry analysis"""
        # Initialize aggregators
        agg_percentiles = None
        agg_exp_curves = None
        agg_equity = None
        
        # Combine group counts
        combined_counts = defaultdict(int)
        
        # Calculate total employees per role for weighting
        role_totals = defaultdict(int)
        for data in self.company_data:
            for role, count in data['group_counts'].items():
                role_totals[role] += count
                combined_counts[role] += count
        
        # Aggregate metrics with proper weighting
        for data in self.company_data:
            # Calculate weight for each role based on company's proportion of total
            weights = {
                role: data['group_counts'][role] / role_totals[role]
                for role in data['group_counts']
                if role_totals[role] > 0  # Avoid division by zero
            }
            
            # Apply weights to encrypted metrics
            weighted_percentiles = data['encrypted_percentiles'] * weights['Software Engineer']  # Assuming same order as metrics
            weighted_exp_curves = data['encrypted_exp_curves'] * weights['Software Engineer']    # May need to adjust per role
            weighted_equity = data['encrypted_equity'] * weights['Software Engineer']            # May need separate weights
            
            # Aggregate weighted metrics
            if agg_percentiles is None:
                agg_percentiles = weighted_percentiles
                agg_exp_curves = weighted_exp_curves
                agg_equity = weighted_equity
            else:
                agg_percentiles += weighted_percentiles
                agg_exp_curves += weighted_exp_curves
                agg_equity += weighted_equity
        
        return EncryptedAnalysis(
            encrypted_percentiles=agg_percentiles,
            encrypted_experience_curves=agg_exp_curves,
            encrypted_pay_gaps=agg_equity,
            group_counts=dict(combined_counts),
            context=self.context
        )

def main():
    # Initialize homomorphic encryption context
    public_context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    public_context.global_scale = 2**40
    public_context.generate_galois_keys()
    
    # Initialize industry analyzer
    analyzer = IndustryAnalyzer(public_context)
    
    # Initialize companies with real data
    companies = {
        'Accenture': Company('Accenture', 'datasets/accenture_employee_salary_data.csv', public_context),
        'GovTech': Company('GovTech', 'datasets/govtech_employee_salary_data.csv', public_context),
        'Tiktok': Company('Tiktok', 'datasets/tiktok_employee_salary_data.csv', public_context)
    }
    
    print("\nSecurely analyzing compensation data...")
    # Companies encrypt and contribute their data
    for name, company in companies.items():
        encrypted_data = company.encrypt_analysis()
        analyzer.add_company_data(name, encrypted_data)
        print(f"✓ {name} data securely processed")
    
    # Perform industry-wide analysis
    print("\nGenerating privacy-protected insights...")
    industry_analysis = analyzer.analyze()
    
    # Generate company-specific insights
    for name, company in companies.items():
        print(f"\nMarket Analysis for {name}:")
        insights = MarketAnalysis(
            name, 
            company.raw_data,  # Pass the company's raw data
            industry_analysis, 
            company.feature_names
        ).get_market_insights()
        
        print("\nMarket Position:")
        for position in insights['market_position']:
            print(f"  • {position}")
        
        print("\nCompensation Drivers:")
        for driver in insights['compensation_drivers']:
            print(f"  • {driver}")
        
        print("\nEquity Analysis:")
        for metric in insights['equity_analysis']:
            print(f"  • {metric}")

if __name__ == "__main__":
    main()