# Usa un'immagine base Python leggera
FROM python:3.8-slim

# Imposta la directory di lavoro nel container
WORKDIR /app

# Copia i file necessari nel container
COPY train.py /app/train.py
COPY requirements.txt /app/requirements.txt

# Installa le dipendenze
RUN pip install --no-cache-dir -r requirements.txt

# Imposta il comando di esecuzione del container
ENTRYPOINT ["python", "/app/train.py"]
