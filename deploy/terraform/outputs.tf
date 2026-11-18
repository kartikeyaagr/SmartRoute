output "cluster_name" {
  value = google_container_cluster.smartroute.name
}

output "image_repository" {
  description = "Set as image.repository in deploy/helm/values-gke.yaml"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}/smartroute"
}

output "ingress_ip" {
  value = google_compute_global_address.ingress.address
}

output "workload_identity_provider" {
  description = "For google-github-actions/auth: workload_identity_provider"
  value       = google_iam_workload_identity_pool_provider.github.name
}

output "deployer_service_account" {
  description = "For google-github-actions/auth: service_account"
  value       = google_service_account.deployer.email
}
