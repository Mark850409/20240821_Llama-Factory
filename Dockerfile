FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY . /app/
RUN pip install -e .[deepspeed,metrics,bitsandbytes,qwen]

EXPOSE 7860

# 設定啟動指令來運行 src/webui.py
CMD ["python", "src/webui.py"]