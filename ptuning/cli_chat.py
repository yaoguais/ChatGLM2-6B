from transformers import AutoTokenizer, AutoModel
import os
import torch
from transformers import AutoConfig

model_path = "THUDM/chatglm2-6b"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def display_answer(model, query, history=[]):
    for response, history in model.stream_chat(
            tokenizer, query, history=history):
        continue
    print("用户: ", query, "\n\n")
    print("ChatGLM: ", response, "\n\n")
    print("End\n\n")
    return history

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, pre_seq_len=128)
model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True)
ptuning_model_path = os.path.join("./output/minsamples-chatglm2-6b-pt--/checkpoint-100", "pytorch_model.bin")
prefix_state_dict = torch.load(ptuning_model_path)
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

model = model.half().cuda()
model.transformer.prefix_encoder.float()
model = model.eval()

display_answer(model, "我希望你扮演一个男性虚拟角色西奥多。我会和你进行聊天，请直接回复我，下面是我说的话。[[对话开始]][[段落开始]]你觉得苏轼怎么样？[[段落结束]]")

