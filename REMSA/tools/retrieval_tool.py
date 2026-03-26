from langchain_core.tools import BaseTool
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import PrivateAttr
import numpy as np
from REMSA.config import config
import json
import os
import duckdb

class FMDRetrievalTool(BaseTool):
    name: str = "RetrieveModels"
    description: str = "Retrieve relevant foundation models based on query using cosine similarity filtering."

    _vectorstore: FAISS = PrivateAttr()
    _embedding: HuggingFaceEmbeddings = PrivateAttr()
    _db: duckdb.DuckDBPyConnection = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Init embedding model
        self._embedding = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        # Load FAISS vector store
        if os.path.exists(config["VECTOR_INDEX_PATH"]):
            self._vectorstore = FAISS.load_local(
                config["VECTOR_INDEX_PATH"],
                self._embedding,
                allow_dangerous_deserialization=True
            )
        else:
            print("FAISS index not found. Initializing empty store.")
            print("Initializing empty vector index.")
            self._vectorstore = FAISS.from_texts([], self._embedding)

        # persistent DB connection
        self._db = duckdb.connect(config["DUCKDB_PATH"], read_only=True)

    def _get_model_metadata(self, model_id: str) -> dict:
        row = self._db.execute(
            #"SELECT * FROM models_top_level_schema WHERE model_id = ?", [model_id]
            "SELECT * FROM foundation_models WHERE model_id = ?", [model_id]
        ).fetchone()
        if not row:
            return {}
        cols = [c[0] for c in self._db.description]
        return {cols[i]: row[i] for i in range(len(cols))}

    def _run(self, query: str | dict):
        try:
            # If it's a dict, serialize it to a string for embedding
            if isinstance(query, dict):
                query = json.dumps(query, indent=2)

            # Embed and normalize query
            query_emb = self._embedding.embed_query(query)
            query_emb = np.array(query_emb)
            query_emb = query_emb / np.linalg.norm(query_emb)

            # Retrieve many candidates (no hard cutoff)
            docs_and_scores = self._vectorstore.similarity_search_with_score(query, k=config["MAX_RETRIEVE"])

            results = []
            tmp_list = []
            for doc, score in docs_and_scores:
                # FAISS uses inner product; normalize to interpret as cosine
                doc_emb = self._embedding.embed_query(doc.page_content)
                doc_emb = np.array(doc_emb)
                doc_emb = doc_emb / np.linalg.norm(doc_emb)
                cosine_sim = float(np.dot(query_emb, doc_emb))


                if cosine_sim >= config["SIMILARITY_BOUNDARY"]:
                    model_id = doc.metadata.get("model_id", "")
                    model_info = self._get_model_metadata(model_id)

                    # fix JSON serialization of dates
                    for k, v in model_info.items():
                        if hasattr(v, "isoformat"):
                            model_info[k] = v.isoformat()

                    model_info["similarity"] = round(cosine_sim, 4)
                    if cosine_sim >= config["MIN_SIMILARITY"]:
                        results.append(model_info)
                    else:
                        tmp_list.append(model_info)

            if len(results) < config["TOP_K"]:
                tmp_list.sort(key=lambda m: m.get("similarity", 0), reverse=True)
                for r in tmp_list:
                    results.append(r)
                    if len(results) >= config["TOP_K"]:
                        break
            return {
                "query": query,
                "candidates": results
            }

        except Exception as e:
            print("[Vector Store Error]", e)
            return {"error": f"Vector store error: {str(e)}"}