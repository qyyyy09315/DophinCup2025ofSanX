from transformers import BertTokenizer, BertModel

# 加载中文BERT的tokenizer和模型
local_model_path = "./bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(local_model_path)
model = BertModel.from_pretrained(local_model_path)

# 示例：对中文文本进行编码
text = "你好，世界！"
inputs = tokenizer(text, return_tensors="pt")  # 返回PyTorch张量
outputs = model(**inputs)

# 获取最后一层的隐藏状态
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape)  # [batch_size, sequence_length, hidden_size]