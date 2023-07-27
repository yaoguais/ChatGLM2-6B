# FineTune

```
!apt update && apt install -y git git-lfs
!git lfs install
!git clone https://ghproxy.com/https://github.com/yaoguais/ChatGLM2-6B.git
!cd ChatGLM2-6B && pip install -r requirements.txt -i https://mirror.sjtu.edu.cn/pypi/web/simple/ && pip install tensorrt
!wget https://cdn-media.huggingface.co/frpc-gradio-0.2/frpc_linux_amd64 && chmod +x frpc_linux_amd64 && mv frpc_linux_amd64 /usr/local/lib/python3.10/site-packages/gradio/frpc_linux_amd64_v0.2
!cd ChatGLM2-6B && python web_demo.py
!pip install rouge_chinese nltk jieba datasets transformers[torch] -i https://mirror.sjtu.edu.cn/pypi/web/simple/
!apt install -y nvidia-cuda-toolkit
!cd ChatGLM2-6B/ptuning && bash ./train_chat.sh
!cd ChatGLM2-6B/ptuning && bash ./web_demo.sh
```
