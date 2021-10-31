import pickle

with open('rf_model.bin', 'rb') as f_in:
    dv, rf_model = pickle.load(f_in)

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

X = dv.transform([employee])
y_pred = rf_model.predict_proba(X)[0, 1]
emp_quit = y_pred >= 0.5

print('The probability of this employee quitting is {0}'.format(y_pred))
print('The employee is likely to quit: {0}'.format(emp_quit))
