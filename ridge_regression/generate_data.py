import numpy as np
import pandas as pd
import os

# Define company-specific salary parameters
COMPANY_SALARY_PARAMS = {
    'Accenture': {
        'Software Engineer': {
            'reference_min': 3000,             # Minimum base salary
            'reference_max': 5000,             # Maximum base salary
            'experience_coef': 500,            # $500 per year of experience
            'education_bachelor_coef': 1500,   # $1,500 bonus for Bachelor's degree
            'education_master_coef': 3000,     # $3,000 bonus for Master's degree
            'gender_bonus': 1000,              # $1,000 bonus for males
            'noise_std': 400                    # Standard deviation for Gaussian noise
        },
        'Data Scientist': {
            'reference_min': 4750,
            'reference_max': 8250,
            'experience_coef': 600,
            'education_bachelor_coef': 2000,
            'education_master_coef': 4000,
            'gender_bonus': 1200,
            'noise_std': 500
        },
        'Product Manager': {
            'reference_min': 5000,
            'reference_max': 8750,
            'experience_coef': 550,
            'education_bachelor_coef': 1800,
            'education_master_coef': 3500,
            'gender_bonus': 1100,
            'noise_std': 450
        }
    },
    'Tiktok': {
        'Software Engineer': {
            'reference_min': 8000,
            'reference_max': 12000,
            'experience_coef': 700,
            'education_bachelor_coef': 2500,
            'education_master_coef': 5000,
            'gender_bonus': 1300,
            'noise_std': 600
        },
        'Data Scientist': {
            'reference_min': 7000,
            'reference_max': 11000,
            'experience_coef': 650,
            'education_bachelor_coef': 2200,
            'education_master_coef': 4500,
            'gender_bonus': 1250,
            'noise_std': 550
        },
        'Product Manager': {
            'reference_min': 7000,
            'reference_max': 12000,
            'experience_coef': 600,
            'education_bachelor_coef': 2000,
            'education_master_coef': 4000,
            'gender_bonus': 1200,
            'noise_std': 500
        }
    },
    'GovTech': {
        'Software Engineer': {
            'reference_min': 6000,
            'reference_max': 9000,
            'experience_coef': 500,
            'education_bachelor_coef': 1800,
            'education_master_coef': 3500,
            'gender_bonus': 1000,
            'noise_std': 500
        },
        'Data Scientist': {
            'reference_min': 7000,
            'reference_max': 11000,
            'experience_coef': 600,
            'education_bachelor_coef': 2000,
            'education_master_coef': 4000,
            'gender_bonus': 1100,
            'noise_std': 550
        },
        'Product Manager': {
            'reference_min': 7000,
            'reference_max': 10000,
            'experience_coef': 550,
            'education_bachelor_coef': 1900,
            'education_master_coef': 3600,
            'gender_bonus': 1050,
            'noise_std': 500
        }
    }
}

def generate_salary_data(
    company_name: str,
    job_title: str,
    params: dict,
    num_employees: int,
    male_percentage: float = 0.5,
    start_id: int = 1  # Added parameter for starting ID
) -> pd.DataFrame:
    """
    Generate employee salary data for a specific company and job title using an additive model.
    
    Args:
        company_name (str): Name of the company.
        job_title (str): Job title.
        params (dict): Salary parameters specific to the company and job title.
        num_employees (int): Number of employee records to generate.
        male_percentage (float): Proportion of male employees.
        start_id (int): Starting index for Employee_ID generation
    Returns:
        pd.DataFrame: Generated salary data.
    """
    # Unpack parameters
    reference_min = params['reference_min']
    reference_max = params['reference_max']
    experience_coef = params['experience_coef']
    education_bachelor_coef = params['education_bachelor_coef']
    education_master_coef = params['education_master_coef']
    gender_bonus = params['gender_bonus']
    noise_std = params['noise_std']
    
    # Generate gender
    gender = np.random.choice(['Male', 'Female'], size=num_employees, p=[male_percentage, 1 - male_percentage])
    
    # Generate age (21-29 for junior positions)
    base_age = np.random.randint(21, 30, num_employees)
    age = np.where(
        gender == 'Male',
        np.clip(base_age + 2, 21, 29),  # Males tend to be 2 years older due to NS
        base_age
    )
    
    # Years of experience (0-4 for junior positions)
    grad_age = np.where(
        gender == 'Male',
        np.random.randint(23, 26, num_employees),  # Males graduate later due to NS
        np.random.randint(21, 24, num_employees)   # Females graduate earlier
    )
    
    civilian_exp = np.clip(age - grad_age, 0, 4)
    ns_credit = np.where(gender == 'Male', 0.5, 0)  # Count 0.5 years of NS as relevant experience
    yoe = np.clip(civilian_exp + ns_credit, 0, 4)    # Cap at 4 years for junior positions
    
    # Tenure: 0-2 for junior positions
    tenure = np.clip(np.random.randint(0, yoe + 1, num_employees), 0, 2)
    
    # Education levels with SG market context
    education_probs = {
        'Software Engineer': {
            'Diploma': 0.20,
            "Bachelor's": 0.70,
            "Master's": 0.10
        },
        'Data Scientist': {
            'Diploma': 0.10,
            "Bachelor's": 0.70,
            "Master's": 0.20
        },
        'Product Manager': {
            'Diploma': 0.15,
            "Bachelor's": 0.70,
            "Master's": 0.15
        }
    }[job_title]
    
    education = np.random.choice(
        list(education_probs.keys()),
        size=num_employees,
        p=list(education_probs.values())
    )
    
    # Calculate additive salary
    final_salary = (
        reference_min +
        (yoe * experience_coef) +
        np.where(education == "Bachelor's", education_bachelor_coef, 0) +
        np.where(education == "Master's", education_master_coef, 0) +
        np.where(gender == 'Male', gender_bonus, 0) +
        np.random.normal(0, noise_std, num_employees)  # Add Gaussian noise
    )
    
    # Ensure salaries stay within reference range with minimal clipping
    final_salary = np.clip(final_salary, reference_min * 0.95, reference_max * 1.05)
    
    # Create DataFrame with simplified Employee_ID
    df = pd.DataFrame({
        'Employee_ID': [f'{job_title}_{i}' for i in range(start_id, start_id + num_employees)],
        'Job_Title': job_title,
        'Gender': gender,
        'Age': age,
        'Education_Level': education,
        'Years_of_Experience': yoe,
        'Years_of_Tenure': tenure,
        'Salary': final_salary.round(2),
        'Company': company_name  # Add company as a regular column
    })
    
    # Reorder columns to match desired format
    column_order = ['Employee_ID', 'Job_Title', 'Gender', 'Age', 'Education_Level', 
                   'Years_of_Experience', 'Years_of_Tenure', 'Salary', 'Company']
    df = df[column_order]
    
    return df

def main():
    # Define job titles
    job_titles = ['Software Engineer', 'Data Scientist', 'Product Manager']
    
    # Initialize list to hold all data
    all_salary_data = []
    
    # Initialize counter for continuous Employee_IDs across companies
    id_counter = 1
    
    # Define number of employees per job title per company
    NUM_EMPLOYEES_PER_ROLE = {
        'Accenture': {
            'Software Engineer': 100,
            'Data Scientist': 50,
            'Product Manager': 30
        },
        'GovTech': {
            'Software Engineer': 100,
            'Data Scientist': 50,
            'Product Manager': 30
        },
        'Tiktok': {
            'Software Engineer': 100,
            'Data Scientist': 50,
            'Product Manager': 30
        }
    }
    
    # Generate data for each company and job title
    for company, roles in COMPANY_SALARY_PARAMS.items():
        for job_title, params in roles.items():
            num_employees = NUM_EMPLOYEES_PER_ROLE[company][job_title]
            df = generate_salary_data(
                company_name=company,
                job_title=job_title,
                params=params,
                num_employees=num_employees,
                male_percentage=0.5,  # Adjust if needed per company
                start_id=id_counter
            )
            all_salary_data.append(df)
            id_counter += num_employees
    
    # Combine all data
    final_data = pd.concat(all_salary_data, ignore_index=True)
    
    # Format salary as currency
    final_data['Salary'] = final_data['Salary'].apply(lambda x: f'${x:,.2f}')
    
    # Ensure the 'data' directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV per company
    for company in COMPANY_SALARY_PARAMS.keys():
        company_data = final_data[final_data['Company'] == company]
        company_data.to_csv(f'data/{company.lower()}_employee_salary_data.csv', index=False)
    
    # Analysis output
    print("Data saved to respective company CSV files in the 'data' directory.")
    print("\nSample Data:")
    print(final_data.head())
    print("\nSummary Statistics by Company and Job Title:")
    # Convert Salary back to float for description
    final_data_numeric = final_data.copy()
    final_data_numeric['Salary'] = final_data_numeric['Salary'].str.replace('[\$,]', '', regex=True).astype(float)
    summary_stats = final_data_numeric.groupby(['Company', 'Job_Title'])['Salary'].describe()
    print(summary_stats)

if __name__ == "__main__":
    main()