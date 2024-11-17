from sentence_transformers import SentenceTransformer
import pandas as pd

# 文件路径
sourceDataPath = "../dataSet/cleaned_sms_data.csv"
finalDataPath = "../dataSet/vectorized_sms_Data.csv"

# 加载数据
df = pd.read_csv(sourceDataPath)

# 确保数据中包含 "Message" 列
if "Message" not in df.columns:
    raise KeyError("The dataset must contain a 'Message' column.")

# 加载预训练的SBERT模型
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 将短信文本转换为列表，处理嵌入
messages = df["Message"].astype(str).tolist()
embeddings = model.encode(messages)

# 将嵌入转为 DataFrame，并与原始数据拼接
embeddings_df = pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(embeddings.shape[1])])
final_df = pd.concat([df.drop(columns=["Message"]), embeddings_df], axis=1)

# 保存嵌入数据到 CSV 文件
final_df.to_csv(finalDataPath, index=False)
print(f"Vectorized data saved to {finalDataPath}")
