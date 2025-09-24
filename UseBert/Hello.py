import torch
from transformers import BertTokenizer, BertModel

# 加载预训练的BERT模型和对应的tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 示例文本数据集
texts = ['This is the first text.', 'This is the second text.', 'And this is the third one.']

# 对文本进行编码并获取特征
features = []
for text in texts:
    # 对文本进行编码
    encoded_input = tokenizer(text, return_tensors='pt')
    # 获取输出
    output = model(**encoded_input)
    # 获取最后一个隐藏层的输出作为特征
    last_hidden_state = output.last_hidden_state
    # 对序列中的所有token输出进行平均池化，得到整个序列的向量表示
    pooled_output = last_hidden_state.mean(dim=1)
    features.append(pooled_output.detach().numpy())

# 将特征列表转换为numpy数组
features_array = torch.tensor(features).numpy()
print('特征数据集的形状:', features_array.shape)