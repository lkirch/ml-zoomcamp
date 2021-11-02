import requests

url = 'http://localhost:8080/predict'

employee = {"satisfaction_level": 5.8, 
            "last_eval": 0.7, 
            "num_projects": 19,
            "avg_monthly_hrs": 256,
            "time_spent_at_company": 4,
            "work_accident": 0,
            "promotion_last_5_yrs": 0,
            "job_category": "sales",
            "salary_group": "low"
            }
response = requests.post(url, json=employee).json()

print(response)