from sentence_transformers import SentenceTransformer
import pandas as pd

def sbert(df):
    # 加载预训练的SBERT模型
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # 将短信文本转换为列表，处理嵌入
    messages = df["Message"].astype(str).tolist()
    embeddings = model.encode(messages)

    # 将嵌入转为 DataFrame，并与原始数据拼接
    embeddings_df = pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(embeddings.shape[1])])
    final_df = pd.concat([df.drop(columns=["Message"]), embeddings_df], axis=1)
    return final_df


