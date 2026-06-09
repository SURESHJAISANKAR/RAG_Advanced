# ✅ Base image (Python runtime)
FROM python:3.10

# ✅ Set working directory inside container
WORKDIR /app

# ✅ Copy all project files into container
COPY . .

# ✅ Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Expose port
EXPOSE 8000

# ✅ Run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

'''Need to Execute this Line 
docker build -t rag-app .
docker run -p 8000:8000 rag-app
'''