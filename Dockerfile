FROM python:3.9

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Create directories
RUN mkdir -p models/models_results fe_data

# Copy both requirements.txt and the rest of the application files
COPY . .

# Install dependencies
RUN pip install spacy
RUN python -m spacy download el_core_news_sm

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

# Command to run the Flask app
CMD ["python", "app.py"]

EXPOSE 5000




