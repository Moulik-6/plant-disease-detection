# AWS GPU Training Instance Setup & Execution Guide

This document outlines the end-to-end setup for provisioning an AWS GPU instance to execute the 9-experiment training queue (`scripts/run_queue.sh`) and sync artifacts to Amazon S3.

---

## 1. Instance Selection & Service Quota

### Recommended Instances
| Instance Type | GPU / Accelerator | VRAM | vCPU | RAM | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`g5.xlarge`** *(Recommended)* | 1x NVIDIA A10G Tensor Core | 24 GB | 4 | 16 GB | Cost-effective, high FP16/BF16 throughput |
| **`g6.xlarge`** | 1x NVIDIA L4 Tensor Core | 24 GB | 4 | 16 GB | Modern Ada Lovelace architecture |
| **`g4dn.xlarge`** *(Budget)* | 1x NVIDIA T4 | 16 GB | 4 | 16 GB | Slower, lower cost fallback |

### Requesting EC2 Service Quota
New AWS accounts have 0 vCPUs allocated for "Running On-Demand G and VT instances":
1. Open the [AWS Service Quotas Console](https://console.aws.amazon.com/servicequotas/).
2. Navigate to **AWS Services** -> **Amazon Elastic Compute Cloud (Amazon EC2)**.
3. Search for **Running On-Demand G and VT instances**.
4. Click **Request quota increase**.
5. Set requested value to at least **4** (for 1x `g5.xlarge` or `g6.xlarge` instance) or **8**.
6. Wait for approval notification (typically 15 minutes to several hours).

---

## 2. Amazon Machine Image (AMI) & Storage

- **Base AMI**: **Deep Learning AMI GPU PyTorch 2.X (Ubuntu 22.04)** (e.g., `ami-xxxxxxxxxxxxxxxxx` in your selected region).
  - Pre-installed with NVIDIA drivers, CUDA 12.x, cuDNN, PyTorch, torchvision, and AWS CLI v2.
- **Root EBS Volume**:
  - Size: **100 GB - 150 GB** (to store raw datasets, image cache, checkpoints, and system packages).
  - Volume Type: **gp3** (3,000 IOPS, 125 MB/s throughput baseline).

---

## 3. Security Group Configuration

Restrict inbound access to your specific management IP address to prevent brute-force attacks:

| Type | Protocol | Port Range | Source | Description |
| :--- | :--- | :--- | :--- | :--- |
| **SSH** | TCP | 22 | `<your-public-ip>/32` | Administrative SSH access only from your IP |

> [!CAUTION]
> Never set the SSH inbound CIDR rule to `0.0.0.0/0`.

---

## 4. IAM Role & S3 Bucket Policy

Attach an IAM Role with an instance profile to the EC2 instance so the AWS CLI can sync to S3 without embedding credentials on disk.

### 1. Create S3 Bucket
```bash
aws s3 mb s3://<your-bucket-name> --region <your-region>
```

### 2. IAM Policy Definition
Create an IAM policy named `PlantDiseaseS3SyncPolicy` scoped strictly to your bucket:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowS3BucketAccess",
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket",
        "s3:GetBucketLocation"
      ],
      "Resource": "arn:aws:s3:::<your-bucket-name>"
    },
    {
      "Sid": "AllowS3ObjectAccess",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::<your-bucket-name>/*"
    }
  ]
}
```

### 3. Attach Role to EC2
1. Create role `PlantDiseaseTrainingRole` with trusted entity `ec2.amazonaws.com`.
2. Attach `PlantDiseaseS3SyncPolicy`.
3. In EC2 Console: **Actions** -> **Security** -> **Modify IAM Role** -> select `PlantDiseaseTrainingRole`.

---

## 5. Provisioning & Environment Setup

Connect to your EC2 instance via SSH:
```bash
ssh -i /path/to/<your-key>.pem ubuntu@<instance-public-ip>
```

### Clone Repository & Activate PyTorch Environment:
```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd plant-disease-detection

# Activate PyTorch virtual environment (Deep Learning AMI default)
source activate pytorch

# Or create a dedicated venv
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r ml/requirements.txt
```

### Download or Sync Dataset:
```bash
# Example: Download dataset from Kaggle or sync from S3
aws s3 sync s3://<your-bucket-name>/data/ data/
```

### Verify Setup:
```bash
python ml/train.py --self-check
```

---

## 6. Running the Experiment Queue

Run the queue inside a detached `tmux` or `screen` session so disconnection does not interrupt training.

```bash
# Start a tmux session
tmux new -s queue

# Configure environment variables
export S3_BUCKET="<your-bucket-name>"
export PER_RUN_TIMEOUT="4h"     # Default per-run execution timeout
export SHUTDOWN="1"              # Automatically shutdown instance upon completion

# Make script executable and launch
chmod +x scripts/run_queue.sh
./scripts/run_queue.sh
```

Detach from the session anytime with `Ctrl+B`, then `D`. Reattach with:
```bash
tmux attach -t queue
```

---

## 7. Safe Stop & Termination Procedures

- **Auto-Shutdown**: When `SHUTDOWN=1` is specified, `scripts/run_queue.sh` will issue `sudo shutdown -h now` upon completion or exit trap.
- **EBS Costs**: Note that a **stopped** instance halts EC2 compute charges, but continues to incur EBS storage costs (~$0.08/GB-month for gp3).
- **Full Termination**: Once experiments have synced to S3, terminate the EC2 instance via AWS Console or CLI:
  ```bash
  aws ec2 terminate-instances --instance-ids <instance-id>
  ```
