FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install openai pydantic fastapi uvicorn python-dotenv

CMD ["python", "inference.py"]