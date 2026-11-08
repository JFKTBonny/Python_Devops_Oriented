# Final Project: AI Support-Ticket Triage Service

Build and deploy a service that analyzes support tickets using a public or
synthetic dataset. Combine the skills from the other AI Foundations folders
with the DevOps practices in this repository.

## Project milestones

1. Use NumPy and Pandas to inspect, clean, and prepare ticket data.
2. Establish a statistical baseline and train and evaluate a simple classifier.
3. Integrate a model API or framework to summarize tickets or suggest a
   category, and compare its output with the baseline.
4. Expose the workflow through a tested Python API; validate inputs and handle
   model failures explicitly.
5. Containerize the service, add CI checks, and deploy it using an appropriate
   cloud platform.
6. Monitor latency, errors, model usage, and prediction quality.

Use synthetic or public data; do not include real customer information or API
credentials in the project.

## Run locally

From this directory, install the shared dependencies and train the model:

```sh
python -m pip install -r ../requirements-dev.txt
python train_model.py
uvicorn app:app --reload
```

Send a ticket to `POST /triage` as JSON, for example:

```json
{"text": "I was charged twice for my monthly subscription"}
```

Run the API tests after training with `python -m pytest`. Build the container
from this directory with `docker build -f Dockerfile -t ticket-triage ..`;
the image installs runtime dependencies and trains the bundled sample model.
The tiny synthetic dataset is for learning and does not indicate real-world
model performance.
