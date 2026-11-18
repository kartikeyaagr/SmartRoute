variable "project_id" {
  description = "GCP project to deploy into (must have billing enabled)"
  type        = string
}

variable "region" {
  description = "Region for the cluster, registry and static IP"
  type        = string
  default     = "us-central1"
}

variable "github_repo" {
  description = "owner/name of the GitHub repo allowed to deploy via Workload Identity Federation"
  type        = string
  default     = "kartikeyaagr/SmartRoute"
}

variable "deletion_protection" {
  description = "Refuse `terraform destroy` on the cluster. Set false to tear a dev cluster down."
  type        = bool
  default     = true
}
