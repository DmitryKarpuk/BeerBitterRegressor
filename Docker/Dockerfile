FROM python:3.9.12
RUN pip install --upgrade pip
RUN pip install poetry
WORKDIR /app
COPY ["./models/model.cbn", "./"]
COPY ["./poetry.lock", "./pyproject.toml", "./"]
COPY ["./src/beerbitterregressor/app/app.py", "./"]
COPY ["./src/beerbitterregressor/preprocessing.py", "./"]
RUN poetry config virtualenvs.create false \
    && poetry install --without dev --no-interaction --no-ansi
EXPOSE 9696
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "app:app"] 
