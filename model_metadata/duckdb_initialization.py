from datetime import datetime
import duckdb
import json

jsonl_path = "../model_metadata/fmd.jsonl"
db_path = "../model_metadata/model_metadata.duckdb"

con = duckdb.connect(db_path)

# write jsonl into DuckDB
con.execute("DROP TABLE IF EXISTS foundation_models")
con.execute("""
CREATE TABLE foundation_models (
    model_id TEXT PRIMARY KEY,
    model_name TEXT,
    version TEXT,
    release_date DATE,
    last_updated DATE,
    short_description TEXT,
    paper_link TEXT,
    citations BIGINT,
    repository TEXT,
    weights TEXT,
    backbone TEXT,
    num_layers BIGINT,
    num_parameters BIGINT,
    pretext_training_type TEXT,
    masking_strategy TEXT,
    pretraining TEXT,
    domain_knowledge JSON,
    backbone_modifications JSON,
    supported_sensors JSON,
    modality_integration_type TEXT,
    modalities JSON,
    spectral_alignment TEXT,
    temporal_alignment TEXT,
    spatial_resolution TEXT,
    temporal_resolution TEXT,
    bands JSON,
    pretraining_phases JSON,
    benchmarks JSON
);
""")

def parse_date(d):
    if not d:
        return None
    return datetime.strptime(d, "%Y-%m-%d").date()

with open(jsonl_path) as f:
    for line in f:
        e = json.loads(line)

        con.execute("""
            INSERT INTO foundation_models VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
        """, [
            e.get("model_id"),
            e.get("model_name"),
            e.get("version"),
            parse_date(e.get("release_date")),
            parse_date(e.get("last_updated")),
            e.get("short_description"),
            e.get("paper_link"),
            e.get("citations"),
            e.get("repository"),
            e.get("weights"),
            e.get("backbone"),
            e.get("num_layers"),
            e.get("num_parameters"),
            e.get("pretext_training_type"),
            e.get("masking_strategy"),
            e.get("pretraining"),
            json.dumps(e.get("domain_knowledge")),
            json.dumps(e.get("backbone_modifications")),
            json.dumps(e.get("supported_sensors")),
            e.get("modality_integration_type"),
            json.dumps(e.get("modalities")),
            e.get("spectral_alignment"),
            e.get("temporal_alignment"),
            e.get("spatial_resolution"),
            e.get("temporal_resolution"),
            json.dumps(e.get("bands")),
            json.dumps(e.get("pretraining_phases")),
            json.dumps(e.get("benchmarks"))
        ])

con.close()
print(f"Wrote models into {db_path}")