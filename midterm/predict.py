import pickle
import numpy

from flask import Flask, request, jsonify

def predict_quitting(employee, dv, model):
    X = dv.transform([employee])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]


with open('rf_model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('predict')

@app.route('/predict', methods=['POST'])
def predict():
    employee = request.get_json()

    y_pred = predict_quitting(employee, dv, model)
    emp_quit = y_pred >= 0.5

    result = {
        'quit_probability': float(y_pred),
        'quit': bool(emp_quit)
    }

    return jsonify(result)

@app.route('/test',methods = ['GET'])
def test():
    return 'test'

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

