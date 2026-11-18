# GKE Autopilot: Google runs the nodes. You declare pods (CPU/memory requests in the
# Helm chart) and are billed per pod resource; there are no node pools to size or patch.

locals {
  services = [
    "container.googleapis.com",
    "artifactregistry.googleapis.com",
    "iamcredentials.googleapis.com",
    "sts.googleapis.com",
  ]
}

resource "google_project_service" "enabled" {
  for_each           = toset(local.services)
  service            = each.value
  disable_on_destroy = false
}

resource "google_container_cluster" "smartroute" {
  name                = "smartroute"
  location            = var.region # regional: Autopilot clusters are always regional
  enable_autopilot    = true
  deletion_protection = var.deletion_protection

  release_channel {
    channel = "REGULAR"
  }

  depends_on = [google_project_service.enabled]
}

resource "google_artifact_registry_repository" "images" {
  repository_id = "smartroute"
  location      = var.region
  format        = "DOCKER"

  # Keep the 20 newest images; older ones are deleted (storage is billed per GB).
  cleanup_policies {
    id     = "keep-recent"
    action = "KEEP"
    most_recent_versions {
      keep_count = 20
    }
  }

  depends_on = [google_project_service.enabled]
}

# Referenced by the Ingress annotation in deploy/helm/values-gke.yaml.
resource "google_compute_global_address" "ingress" {
  name = "smartroute-ip"
}

# ---------------------------------------------------------------------------
# GitHub Actions → GCP without a JSON key.
# GitHub mints a short-lived OIDC token per workflow run; GCP trusts tokens from
# this one repo and exchanges them for a deployer service account's credentials.
# Nothing long-lived to leak or rotate.
# ---------------------------------------------------------------------------

resource "google_service_account" "deployer" {
  account_id   = "github-deployer"
  display_name = "GitHub Actions deployer for SmartRoute"
}

resource "google_project_iam_member" "deployer" {
  for_each = toset([
    "roles/container.developer",     # helm upgrade into the cluster
    "roles/artifactregistry.writer", # push images
  ])
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.deployer.email}"
}

resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "github"
  depends_on                = [google_project_service.enabled]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-oidc"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
  }
  # Without this, ANY GitHub repo's workflow could impersonate the deployer.
  attribute_condition = "assertion.repository == '${var.github_repo}'"

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

resource "google_service_account_iam_member" "github_impersonates_deployer" {
  service_account_id = google_service_account.deployer.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repo}"
}
