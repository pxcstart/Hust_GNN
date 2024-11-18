from transformers import RobertaTokenizer, RobertaModel, AutoTokenizer, AutoModel
import torch

model_name = '/data2/pxc/CodeLlama-7b-hf'
tokenizer_name = '/data2/pxc/CodeLlama-7b-hf'

# 加载 Tokenizer 和 Model
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token
# model = RobertaModel.from_pretrained(model_name).to(torch.device('cuda'))
model = AutoModel.from_pretrained(model_name, device_map="auto")

def get_code_embedding(code):
    inputs = tokenizer(code, return_tensors='pt', truncation=True, padding=True, max_length=512).to(torch.device('cuda'))
    with torch.no_grad():
        outputs = model(**inputs)
    code_embedding = outputs.last_hidden_state.mean(dim=1)  

    return code_embedding
