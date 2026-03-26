import json
import duckdb
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import yaml

with open("/config.yaml") as f:
    config = yaml.safe_load(f)

def flatten(entry, prefix=""):
    flattened = []
    if isinstance(entry, dict):
        for k, v in entry.items():
            key = f"{prefix}{k}" if prefix == "" else f"{prefix}.{k}"
            flattened.extend(flatten(v, prefix=key))
    elif isinstance(entry, list):
        for i, item in enumerate(entry):
            key = f"{prefix}[{i}]"
            flattened.extend(flatten(item, prefix=key))
    else:
        flattened.append(f"{prefix}: {entry}")
    return flattened

def build_vectorstore():
    print("Building FAISS vector store from FMD...")

    # with open(config["FMD_JSONL_PATH"], 'r') as f:
    #     entries = [json.loads(line) for line in f]
    con = duckdb.connect(config["DUCKDB_PATH"], read_only=True)
    #rows = con.execute("SELECT json FROM models").fetchall()
    #entries = [json.loads(r[0]) for r in rows]

    #rows_2 = con.execute("SELECT * FROM models_top_level_schema").fetchall()
    rows_2 = con.execute("SELECT * FROM foundation_models").fetchall()
    cols_2 = [c[0] for c in con.description]
    entries_2 = [dict(zip(cols_2, row)) for row in rows_2]

    #print("ENTIRES")
    #print(entries[0])

    #print("\n ENTRIES")
    #print(entries_2[0])

    json_fields = [
        "domain_knowledge",
        "backbone_modifications",
        "supported_sensors",
        "modalities",
        "bands",
        "pretraining_phases",
        "benchmarks",
    ]

    docs = []
    for entry in entries_2:
        # convert JSON strings back to Python lists/dicts
        for field in json_fields:
            if isinstance(entry.get(field), str):
                try:
                    entry[field] = json.loads(entry[field])
                except:
                    pass  # leave as-is if not valid JSON

        flat_text = "\n".join(flatten(entry))
        docs.append(
            Document(
                page_content=flat_text,
                metadata={"model_id": entry["model_id"]}
            )
        )
    # Embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=config["EMBEDDING_MODEL_NAME"],
        model_kwargs={"device": "cpu"}  # or "cpu"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(config["TARGET_DIR"])

    print(f"FAISS index saved to {config['TARGET_DIR']}")

if __name__ == "__main__":
    build_vectorstore()
