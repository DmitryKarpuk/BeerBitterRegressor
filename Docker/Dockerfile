FROM python:3.9.12
RUN pip install --upgrade pip
RUN pip install poetry
COPY ["./models/model.cbn", "app/models"]
COPY ["./poetry.lock", "./pyproject.toml", "./app/"]
COPY ["./src/beerbitterregressor/app/app.py", "./app/"]
WORKDIR /app
RUN poetry config virtualenvs.create false \
    && poetry install --without dev --no-interaction --no-ansi
EXPOSE 9696
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "app:app"] 
