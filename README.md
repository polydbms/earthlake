# EarthLake: Model Lake System for Earth Observation Foundation Model Management

![Alt text](figures/Earthlake.png?raw=true "Architecture of EarthLake")

This repository contains the code of the paper **Demonstrating EarthLake: A Model Lake System for Earth Observation Foundation Model Management**. The work introduces EarthLake, a model lake system for Earth Observation Foundation Models (EO FMs), enabling users to discover, evaluate, compare, and operationalize EO FMs in a unified workflow.

This work has been done at [BIFOLD](https://www.bifold.berlin/) and [TU Berlin](https://www.tu.berlin/) by Binger Chen, Haralampos Gavriilidis, Luca Gaedicke, Tacettin Emre Bök, Matthias Boehm, Ziawasch Abedjan, Begüm Demir, and Volker Markl.

If you use this code, please cite our paper given below:

> B. Chen, H. Gavriilidis, L. Gaedicke, T. E. Bök, M. Boehm, Z. Abedjan, B. Demir, V. Markl, "Demonstrating EarthLake: A Model Lake System for Earth Observation Foundation Model Management", PVLDB demonstration paper, 2026.

```bibtex
@misc{chen2026earthlake,
      title={Demonstrating EarthLake: A Model Lake System for Earth Observation Foundation Model Management},
      author={Binger Chen and Haralampos Gavriilidis and Luca Gaedicke and Tacettin Emre Bök and Matthias Boehm and Ziawasch Abedjan and Begüm Demir and Volker Markl},
      year={2026},
      note={PVLDB demonstration paper},
      url={https://github.com/polydbms/earthlake/},
}
```

---

## Overview

**EarthLake** supports end-to-end management of Earth Observation Foundation Models by combining:

- A schema-guided model registry for EO FMs
- Natural-language and SQL-based model discovery
- Integration with REMSA for task-driven model recommendation
- Reproducible benchmarking and model comparison
- Operational inference on new EO imagery

The system enables EO analysts and model developers to manage heterogeneous foundation models, metadata, benchmark configurations, evaluation results, and model artifacts within a unified model lake.

---

## Environment Setup

This project uses Docker Compose for containerized setup.

Create the environment configuration file:

```bash
cp .env.example .env
```

Set your OpenAI API key in the `.env` file.

Run EarthLake with the correct profile. Use `gpu` for CUDA-enabled environments, and `cpu` otherwise, including CPU and MPS-based setups:

```bash
docker compose --profile cpu up
```

For GPU environments:

```bash
docker compose --profile gpu up
```

After startup, open the web interface at:

```bash
http://localhost:5173
```

---

## Configuration

EarthLake is configured through the project environment and Docker Compose setup. The following parameters can be adjusted:

- OpenAI API key
- Runtime profile selection
- Backend and frontend service configuration
- Model registry and metadata paths
- Benchmarking and inference runtime settings

Adjust these parameters according to your local runtime environment and available hardware.

---

## Running EarthLake

To launch EarthLake:

```bash
docker compose --profile cpu up
```

Then navigate to:

```bash
http://localhost:5173
```

EarthLake provides a unified interface for registering models, discovering suitable EO FMs, benchmarking candidate models, comparing evaluation results, and running inference on new imagery.

---

## Data Structure

### Model Registry

- Defined by the structured EO foundation model schema
- Stores metadata for EO FMs
- Captures model capabilities, supported tasks, sensor modalities, architectures, training details, and operational constraints
- Supports browsing, filtering, SQL querying, and natural-language model discovery

### Model Artifacts

- Stores model weights, code, runtime dependencies, and related model resources
- Allows models to be managed as first-class assets in the model lake

### Experiment Database

- Records benchmark configurations, datasets, hardware contexts, and evaluation metrics
- Enables reproducible comparison of candidate models across experiments

### Datasets and Evaluation Outputs

- Stores user-provided datasets and generated benchmark results
- Supports downstream comparison, selection, and operational deployment of EO FMs

---

## Model Registration

EarthLake allows model developers to register new EO foundation models into the model lake.

Model documentation, such as research papers or model cards, can be used to extract structured metadata fields, including:

- Supported EO tasks
- Input data modalities
- Architecture information
- Training details
- Runtime and hardware constraints

The extracted metadata populates the model registry and makes each model searchable and comparable within EarthLake.

---

## Model Discovery

EarthLake supports model discovery through both structured and natural-language interfaces.

Users can search the model lake using:

- SQL queries over the structured model registry
- Natural-language task descriptions
- Operational constraints such as modality, hardware availability, and target task

EarthLake integrates REMSA as its discovery module to retrieve and rank candidate EO FMs. The system returns a top-k list of recommended models together with metadata and explanations for their task compatibility.

---

## Benchmarking and Evaluation

EarthLake enables users to benchmark candidate models on their own EO datasets.

The benchmarking workflow supports:

- Model selection from discovery results
- Dataset upload
- Evaluation configuration
- Linear probing
- Fine-tuning
- Performance comparison across candidate models

Benchmark results are stored in the experiment database, allowing users to revisit, compare, and reuse evaluation evidence across runs.

---

## Model Operation

After evaluation, users can operationalize the selected model directly within EarthLake.

EarthLake loads the validated model configuration and executes inference through a unified runtime interface. Users can upload new satellite imagery and run EO tasks such as classification or segmentation, with prediction results shown in the interface.

This connects model discovery, benchmarking, and deployment in a single workflow.

---

## REMSA Integration

EarthLake uses REMSA for task-driven EO foundation model discovery.

The original implementation of REMSA is available at:

[https://github.com/be-chen/REMSA.git](https://github.com/be-chen/REMSA.git)

---

## Authors

**Binger Chen**  
chen@tu-berlin.de

**Haralampos Gavriilidis**  
gavriilidis@tu-berlin.de

**Luca Gaedicke**  
luca.gaedicke@tu-berlin.de

**Tacettin Emre Bök**  
boek@tu-berlin.de

**Matthias Boehm**  
matthias.boehm@tu-berlin.de

**Ziawasch Abedjan**  
abedjan@tu-berlin.de

**Begüm Demir**  
demir@tu-berlin.de

**Volker Markl**  
volker.markl@tu-berlin.de

For questions, requests and concerns, please contact [Binger Chen](mailto:chen@tu-berlin.de).

## License

The code in this repository is licensed under the terms specified in the `LICENSE` file.
