# Build a Computer Use Agent with Synthetic Data

Generate synthetic training data for a LangGraph CLI tool-calling model using NVIDIA's NeMo Data Designer.

## Prerequisites

- **Docker** and **Docker Compose** installed
- **NGC CLI API key** - Get one at [ngc.nvidia.com](https://ngc.nvidia.com)
- **NGC CLI** installed (`pip install ngc-cli`)
- **Python 3.10+** with Jupyter

## Setup: Launch NeMo Data Designer

Before running the notebook, you need to start the NeMo Data Designer service via Docker.

### 1. Set your NGC API Key

```bash
export NGC_CLI_API_KEY="your-api-key-here"
```

### 2. Authenticate with NGC

```bash
echo $NGC_CLI_API_KEY | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

### 3. Download the Docker Compose Configuration

```bash
ngc registry resource download-version "nvidia/nemo-microservices/nemo-data-designer-docker-compose:25.08"
```

### 4. Start the Data Designer Services

```bash
cd nemo-data-designer-docker-compose_v25.08
docker compose -f docker-compose.ea.yaml up -d
```

The service will start on **port 8000**. Wait a minute or two for all services to initialize.

### 5. Verify the Service is Running

```bash
curl http://localhost:8000/health
```

You should receive a `200 OK` response when the service is ready.

## Running the Notebook

Once the Data Designer service is running, open and run `langgraph_cli_synthetic_data.ipynb` to generate your synthetic dataset.

The notebook will:
1. Connect to the local Data Designer service
2. Define a schema for LangGraph CLI commands
3. Generate synthetic natural language queries and structured JSON outputs
4. Export the dataset to JSONL format for model training

## Cleanup

When you're done, stop the Docker services:

```bash
cd nemo-data-designer-docker-compose_v25.08
docker compose -f docker-compose.ea.yaml down
```

## Output Files

After running the notebook, you'll have:

| File | Description |
|------|-------------|
| `langgraph_cli_synthetic.jsonl` | Full dataset with all columns |
| `langgraph_cli_training.jsonl` | Training pairs (input/output only) |

## LangGraph CLI Commands Covered

| Command | Description |
|---------|-------------|
| `langgraph new` | Create a new LangGraph project from a template |
| `langgraph dev` | Run the development server with hot reload |
| `langgraph up` | Launch the server in a Docker container |
| `langgraph build` | Build the Docker image for deployment |
| `langgraph dockerfile` | Generate a Dockerfile for the project |

