variable "aws_region" {
  type        = string
  description = "Região da AWS para provisionamento"
  default     = "us-east-1"
}

variable "project_name" {
  type        = string
  description = "Nome do projeto para identificação de recursos"
  default     = "fiap-tech-challenge-fase-2"
}

variable "environment" {
  type        = string
  description = "Ambiente de deployment"
  default     = "production"
}

variable "container_image_backend" {
  type        = string
  description = "Imagem Docker para o container do backend FastAPI"
  default     = "tech-challenge-backend:latest"
}

variable "container_image_frontend" {
  type        = string
  description = "Imagem Docker para o container do frontend Streamlit"
  default     = "tech-challenge-frontend:latest"
}

variable "gemini_api_key" {
  type        = string
  description = "Chave de API do Gemini para integração com LLM (opcional se usar Vertex AI)"
  sensitive   = true
  default     = ""
}

variable "use_vertex_ai" {
  type        = bool
  description = "Ativar o uso do Google Cloud Vertex AI em vez do Gemini Developer API"
  default     = false
}

variable "vertex_project" {
  type        = string
  description = "ID do Projeto no Google Cloud Platform (GCP)"
  default     = ""
}

variable "vertex_location" {
  type        = string
  description = "Região/Localização do recurso no Vertex AI"
  default     = "us-central1"
}

variable "gcp_service_account_key" {
  type        = string
  description = "Conteúdo JSON da chave da Conta de Serviço do GCP para autenticação no Vertex AI"
  sensitive   = true
  default     = ""
}
