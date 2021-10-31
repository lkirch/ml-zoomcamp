FROM python:3.8.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict_quitting_serving.py", "--bind", "0.0.0.0:9696", "rf_model.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "quitting:app"]
