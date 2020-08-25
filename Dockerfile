FROM python:3.7.9

WORKDIR /Generic_GNN

COPY requirements.txt ./

RUN pip install

COPY . .

ENV PORT=8000

EXPOSE 8000

CMD ["python", "main.py", "-c", "config\primary_config.json"]