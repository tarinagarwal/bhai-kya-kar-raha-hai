# Terraform Infrastructure for Gemini Computer Use

This directory contains Terraform configuration for deploying the Gemini Computer Use application to Google Cloud Platform.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Google Cloud Platform                     │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              Artifact Registry                          ││
│  │    gemini-computer-use (Docker Repository)             ││
│  └─────────────────────────────────────────────────────────┘│
│                           │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              Cloud Run Service                          ││
│  │    computer-use-preview                                ││
│  │    - 2 CPU, 2GB RAM                                    ││
│  │    - Playwright + Chromium                             ││
│  │    - WebSocket support                                 ││
│  │    - Auto-scaling 0-10 instances                       ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **Terraform** >= 1.0.0
2. **gcloud CLI** authenticated with appropriate permissions
3. **GCP Project** with billing enabled

## Quick Start

### 1. Initialize Terraform

```bash
cd terraform
terraform init
```

### 2. Configure Variables

```bash
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your project ID
```

### 3. Preview Changes

```bash
terraform plan
```

### 4. Apply Infrastructure

```bash
terraform apply
```

### 5. Get Output Values

```bash
terraform output
```

## Configuration Variables

| Variable        | Description               | Default       |
| --------------- | ------------------------- | ------------- |
| `project_id`    | GCP Project ID            | (required)    |
| `region`        | Deployment region         | `us-central1` |
| `cpu`           | CPU allocation            | `2`           |
| `memory`        | Memory allocation         | `2Gi`         |
| `max_instances` | Max Cloud Run instances   | `10`          |
| `timeout`       | Request timeout (seconds) | `3600`        |

## Resources Created

- **Google Cloud APIs**: Cloud Run, Cloud Build, Artifact Registry
- **Artifact Registry**: Docker repository for container images
- **Cloud Run Service**: Serverless container hosting
- **IAM Policy**: Public access for demo purposes

## Outputs

| Output              | Description                      |
| ------------------- | -------------------------------- |
| `service_url`       | HTTP URL of the deployed service |
| `websocket_url`     | WebSocket URL for frontend       |
| `docker_repository` | Artifact Registry URL            |

## Cleanup

To destroy all resources:

```bash
terraform destroy
```

## Cost Estimate

- **Cloud Run**: Pay per use (~$0 when idle)
- **Artifact Registry**: ~$0.10/GB/month storage
- **Cloud Build**: Free tier available

**Note**: This deployment is optimized for hackathon/demo use. For production, consider adding authentication and monitoring.
