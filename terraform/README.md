# 🚀 Guia de Implantação na Nuvem (AWS com Terraform)

Este guia orienta o deploy automatizado da arquitetura resiliente na nuvem **AWS** utilizando **Terraform (IaC)** e tarefas serverless no **AWS Fargate (ECS)**.

---

## 📋 Pré-requisitos
Antes de iniciar, certifique-se de possuir instalado na sua máquina:
1. **AWS CLI** configurado com credenciais válidas de administrador (`aws configure`).
2. **Terraform CLI** (versão `>= 1.5.0`).
3. **Docker** instalado e em execução local para construir e subir as imagens.

---

## 🛠️ Passo a Passo para Implantação

### Passo 1: Autenticar o Docker no AWS ECR
Para enviar as imagens Docker para a nuvem, precisamos criar os repositórios no AWS ECR e logar o Docker local na AWS.
Execute os seguintes comandos no terminal (substituindo `<SUA_CONTA_AWS>` pelo ID de 12 dígitos da sua conta e `<REGIAO>` pela região desejada, ex: `us-east-1`):

```bash
# 1. Autenticar o Docker
aws ecr get-login-password --region <REGIAO> | docker login --username AWS --password-stdin <SUA_CONTA_AWS>.dkr.ecr.<REGIAO>.amazonaws.com

# 2. Criar os repositórios de imagens no ECR (se não existirem)
aws ecr create-repository --repository-name tech-challenge-backend --region <REGIAO>
aws ecr create-repository --repository-name tech-challenge-frontend --region <REGIAO>
```

---

### Passo 2: Construir e Fazer o Push das Imagens Docker
A partir do diretório raiz do projeto, execute o build e o push das imagens rotulando-as com o endereço dos seus repositórios ECR:

```bash
# ==========================================
# BACKEND (FastAPI)
# ==========================================
# 1. Build da imagem (Importante: use --platform linux/amd64 se estiver no Mac Apple Silicon)
docker build --platform linux/amd64 -f backend/Dockerfile -t tech-challenge-backend:latest .

# 2. Tag a imagem para o ECR
docker tag tech-challenge-backend:latest <SUA_CONTA_AWS>.dkr.ecr.<REGIAO>.amazonaws.com/tech-challenge-backend:latest

# 3. Push para o ECR
docker push <SUA_CONTA_AWS>.dkr.ecr.<REGIAO>.amazonaws.com/tech-challenge-backend:latest

# ==========================================
# FRONTEND (Streamlit)
# ==========================================
# 1. Build da imagem (Importante: use --platform linux/amd64 se estiver no Mac Apple Silicon)
docker build --platform linux/amd64 -f frontend/Dockerfile -t tech-challenge-frontend:latest .

# 2. Tag a imagem para o ECR
docker tag tech-challenge-frontend:latest <SUA_CONTA_AWS>.dkr.ecr.<REGIAO>.amazonaws.com/tech-challenge-frontend:latest

# 3. Push para o ECR
docker push <SUA_CONTA_AWS>.dkr.ecr.<REGIAO>.amazonaws.com/tech-challenge-frontend:latest
```

---

### Passo 3: Configurar Variáveis do Terraform (Opções de LLM)
As variáveis já estão definidas no arquivo `variables.tf`. Para configurá-las com os seus valores reais de implantação, crie um arquivo chamado `terraform.tfvars` na pasta `/terraform`:

#### Opção A: Se estiver utilizando a Gemini Developer API (com chave)
```hcl
aws_region               = "us-east-1"
project_name             = "fiap-tech-challenge-fase-2"
environment              = "production"
container_image_backend  = "<SUA_CONTA_AWS>.dkr.ecr.<REGIAO>.amazonaws.com/tech-challenge-backend:latest"
container_image_frontend = "<SUA_CONTA_AWS>.dkr.ecr.<REGIAO>.amazonaws.com/tech-challenge-frontend:latest"
gemini_api_key           = "SUA_CHAVE_API_GEMINI_AQUI"
```

#### Opção B: Se estiver utilizando o Google Vertex AI
Quando a aplicação roda na nuvem AWS Fargate, ela não tem acesso automático ao seu login local (`gcloud auth application-default login`). Para autenticar a aplicação na AWS junto ao GCP, você deve gerar uma chave JSON de Conta de Serviço (Service Account Key) no Console do Google Cloud e configurá-la no `terraform.tfvars`:

```hcl
aws_region               = "us-east-1"
project_name             = "fiap-tech-challenge-fase-2"
environment              = "production"
container_image_backend  = "<SUA_CONTA_AWS>.dkr.ecr.<REGIAO>.amazonaws.com/tech-challenge-backend:latest"
container_image_frontend = "<SUA_CONTA_AWS>.dkr.ecr.<REGIAO>.amazonaws.com/tech-challenge-frontend:latest"

use_vertex_ai            = true
vertex_project           = "ID-DO-SEU-PROJETO-GCP"
vertex_location          = "us-central1"
# Copie e cole todo o conteúdo do arquivo JSON da chave da conta de serviço entre aspas:
gcp_service_account_key  = <<EOF
{
  "type": "service_account",
  "project_id": "...",
  "private_key_id": "...",
  "private_key": "...",
  "client_email": "...",
  ...
}
EOF
```

---

### Passo 4: Inicializar e Aplicar o Terraform
Com os terminais posicionados na pasta `/terraform`, execute os comandos para criar a infraestrutura:

```bash
# 1. Inicializar o Terraform (faz o download dos providers)
terraform init

# 2. Validar a configuração e planejar a infraestrutura
terraform plan

# 3. Aplicar o provisionamento (digite "yes" para confirmar)
terraform apply
```

Ao finalizar a execução (o que costuma levar de 3 a 5 minutos para provisionar VPC, ALB, ECS e Fargate), o console exibirá o output `frontend_url`:

```bash
Outputs:
frontend_url = "http://fiap-tech-challenge-fase-2-alb-123456789.us-east-1.elb.amazonaws.com"
```

Copie este endereço e abra-o no seu navegador para utilizar a aplicaçãoStreamlit executando 100% na nuvem!

---

## 🧹 Limpeza de Recursos (Destruir Infraestrutura)
Para evitar custos desnecessários na AWS após o término dos testes da banca, destrua toda a infraestrutura criada com o comando:

```bash
terraform destroy
```
*(Digite "yes" para confirmar e aguarde o término do processo).*
