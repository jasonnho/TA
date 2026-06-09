FROM python:3.11-slim

WORKDIR /app

# Matplotlib butuh dir cache yang writable di HF Spaces
ENV MPLCONFIGDIR=/tmp/matplotlib

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces (Docker) mengarahkan trafik ke port 7860
EXPOSE 7860

CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "--workers", "1", "app:server"]
