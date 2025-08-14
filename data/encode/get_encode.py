import json
from pathlib import Path
import openai
import math
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv("../../.env")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_type = "openai" 

# ---------- Utility functions ----------
def get_embedding(text: str, model: str = "text-embedding-3-large"):
    """Return embedding vector for a given text"""
    response = openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def flatten_value(val):
    if isinstance(val, list):
        if all(isinstance(x, dict) for x in val):
            return "; ".join(", ".join(f"{k}: {v}" for k, v in d.items()) for d in val)
        else:
            return ", ".join(str(x) for x in val)
    elif isinstance(val, dict):
        return ", ".join(f"{k}: {v}" for k, v in val.items())
    else:
        return str(val)

def schema_to_text(field_name: str, field_info: dict) -> str:
    parts = [f"Field name: {field_name}"]
    parts.append(f"Description: {field_info.get('description', '')}")
    if field_info.get("data_type") is not None:
        parts.append(f"Data type: {flatten_value(field_info['data_type'])}")
    if field_info.get("data_sample") is not None:
        parts.append(f"Sample values: {flatten_value(field_info['data_sample'])}")
    if field_info.get("possible_relationships") is not None:
        parts.append(f"Relationships: {flatten_value(field_info['possible_relationships'])}")
    if field_info.get("possible_formulas") is not None:
        parts.append(f"Possible formulas: {flatten_value(field_info['possible_formulas'])}")
    if field_info.get("domain_knowledge") is not None:
        parts.append(f"Domain knowledge: {flatten_value(field_info['domain_knowledge'])}")
    return "\n".join(parts)

def cosine_similarity(vec1, vec2):
    dot = sum(a*b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a*a for a in vec1))
    norm2 = math.sqrt(sum(b*b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)

# ---------- Function 1: Encode tables ----------
def encode_tables(data_path: str, out_dir: str) -> str:
    """
    Encode all JSON tables in data_path and save embeddings in out_dir.
    Returns the path of the embeddings JSON file.
    """
    data_path = Path(data_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    table_vector_store = []

    for file in sorted(data_path.iterdir()):
        if not (file.suffix == ".json" and file.is_file()):
            continue
        try:
            with open(file, "r", encoding="utf-8") as f:
                schema = json.load(f)

            # Determine schema structure
            if isinstance(schema, dict) and "columns" in schema:
                schema_dict = schema["columns"]
            elif isinstance(schema, list) and schema:
                schema_dict = {k: {"data_type": type(v).__name__, "data_sample": [v]} for k, v in schema[0].items()}
            else:
                continue

            # Flatten all fields into a single text chunk
            table_text_chunks = [schema_to_text(f, info) for f, info in schema_dict.items()]
            table_text = "\n".join(table_text_chunks)

            # Embed the table
            embedding = get_embedding(table_text)
            table_vector_store.append({
                "file": file.name,
                "text": table_text,
                "embedding": embedding
            })
            print(f"✅ Encoded table: {file.name}")
        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

    # Save embeddings to JSON
    embeddings_file = out_dir / "table_encode_embeddings.json"
    with open(embeddings_file, "w", encoding="utf-8") as f:
        json.dump(table_vector_store, f, ensure_ascii=False, indent=2)
    print(f"✅ All table embeddings saved to {embeddings_file}")

    return str(embeddings_file)

# ---------- Function 2: Retrieve most relevant table ----------
def retrieve_table(question: str, embeddings_file_path: str, top_k: int = 1):
    """
    Given a question, automatically load table embeddings from embeddings_file_path
    and return the top_k relevant table names as a comma-separated string.
    """
    # Load embeddings
    with open(embeddings_file_path, "r", encoding="utf-8") as f:
        table_vector_store = json.load(f)

    # Compute similarities
    question_embedding = get_embedding(question)
    scores = [(cosine_similarity(question_embedding, item["embedding"]), item) for item in table_vector_store]
    scores.sort(key=lambda x: x[0], reverse=True)

    # Extract top table names
    top_table_names = [item["file"] for _, item in scores[:top_k]]

    # Return table names as a comma-separated string
    return ", ".join(top_table_names)

# ---------- Example usage ----------
if __name__ == "__main__":
    # encode_tables("dataset/encode", "encode_embeddings")
    embeddings_file_path = "table_info_encode_mapping.json"
    df = pd.read_csv("close_encode.csv")
    df["relevant_table_info"] = None

    for i in range(len(df)):
        question = df.iloc[i]["question"]
        level = df.iloc[i]["level"]
        top_k = 1 if level == "easy" else 3
        relevant_table_info = retrieve_table(question, embeddings_file_path, top_k=top_k)
        df.at[i, "relevant_table_info"] = relevant_table_info
        print(f"Question: {question}\nRelevant tables: {relevant_table_info}\n")
        df.to_csv("close_encode_with_tables.csv", index=False)