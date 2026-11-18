terraform {
  required_version = ">= 1.6"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 7.0"
    }
  }

  # State lives in a GCS bucket you create once by hand (chicken-and-egg: Terraform
  # can't store its state in a bucket it hasn't created yet):
  #   gcloud storage buckets create gs://<project>-tfstate --location=<region>
  #   terraform init -backend-config="bucket=<project>-tfstate"
  backend "gcs" {
    prefix = "smartroute"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
