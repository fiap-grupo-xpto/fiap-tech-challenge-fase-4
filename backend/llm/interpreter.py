import os
import random
import time
from dotenv import load_dotenv
try:
    from google import genai
except ImportError:  # pragma: no cover - depends on optional runtime package
    genai = None

load_dotenv()

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
GEMINI_RETRY_BASE_DELAY = float(os.getenv("GEMINI_RETRY_BASE_DELAY", "1.5"))
_client = None
_client_error = None


def get_client_or_error():
    global _client, _client_error
    if _client is not None:
        return _client, None
    if _client_error is not None:
        return None, _client_error

    if genai is None:
        return None, "Biblioteca 'google-genai' nao instalada."

    use_vertex = os.getenv("USE_VERTEX_AI", "false").lower() in ("true", "1", "yes")

    try:
        if use_vertex:
            gcp_key = os.getenv("GCP_SERVICE_ACCOUNT_KEY")
            if gcp_key:
                key_path = "/tmp/gcp_key.json"
                try:
                    with open(key_path, "w", encoding="utf-8") as f:
                        f.write(gcp_key)
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
                except Exception:
                    pass
            project = os.getenv("VERTEX_PROJECT")
            location = os.getenv("VERTEX_LOCATION", "us-central1")
            _client = genai.Client(vertexai=True, project=project, location=location)
        else:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return None, "GEMINI_API_KEY nao configurada e USE_VERTEX_AI nao ativado"
            _client = genai.Client(api_key=api_key)
        return _client, None
    except Exception as exc:
        _client_error = str(exc)
        return None, _client_error


def classify_risk(probability: float) -> str:
    if probability >= 0.7:
        return "alto"
    if probability >= 0.4:
        return "moderado"
    return "baixo"


def _is_retryable_quota_error(exc: Exception) -> bool:
    message = str(exc).lower()
    retry_markers = (
        "429",
        "quota",
        "resource_exhausted",
        "rate limit",
        "too many requests",
    )
    return any(marker in message for marker in retry_markers)


def _fallback_message(model_name: str, probability: float, risk_level: str, details: str) -> str:
    return (
        f"O modelo {model_name} indicou uma probabilidade estimada de "
        f"{probability * 100:.2f}%, classificada como risco {risk_level}. "
        "Nao foi possivel gerar a interpretacao pela LLM neste momento. "
        f"Detalhes tecnicos: {details}"
    )


def _generate_interpretation(prompt: str, model_name: str, probability: float, risk_level: str) -> str:
    client, error_details = get_client_or_error()
    if client is None:
        return _fallback_message(
            model_name,
            probability,
            risk_level,
            f"cliente LLM indisponivel. Detalhes: {error_details}",
        )

    last_error = None
    try:
        for attempt in range(GEMINI_MAX_RETRIES + 1):
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL_NAME,
                    contents=prompt,
                )
                return response.text
            except Exception as exc:  # pragma: no cover - depends on external API
                last_error = exc
                if attempt >= GEMINI_MAX_RETRIES or not _is_retryable_quota_error(exc):
                    break

                jitter = random.uniform(0.0, 0.25)
                delay = GEMINI_RETRY_BASE_DELAY * (2 ** attempt) + jitter
                time.sleep(delay)
    except Exception as exc:  # pragma: no cover - defensive
        last_error = exc

    return _fallback_message(model_name, probability, risk_level, str(last_error))


def generate_tabular_interpretation(result: dict, model_name: str, threshold: float) -> str:
    probability = result["probability"]
    disease_detected = result["disease_detected"]
    risk_level = classify_risk(probability)

    prompt = f"""
Você é uma LLM integrada a um sistema de apoio ao diagnóstico médico.

Explique o resultado abaixo de forma clara, objetiva e responsável.

Dados do modelo:
- Modelo utilizado: {model_name}
- Tipo de exame: triagem tabular a partir de dados demográficos e sintomas informados pelo paciente
- Probabilidade estimada: {probability * 100:.2f}%
- Limiar de decisão: {threshold * 100:.2f}%
- Doença detectada pelo modelo: {"sim" if disease_detected else "não"}
- Nível de risco estimado: {risk_level}

Responda obrigatoriamente neste formato:

Resumo:
Risco:
Justificativa:
Recomendação:
Observação:

Regras:
- Não diga que o paciente tem diagnóstico confirmado.
- Não substitua avaliação médica.
- Use linguagem adequada para profissionais da saúde.
- Seja objetivo.
- A recomendação deve ser compatível com o nível de risco.
- A observação deve deixar claro que o resultado é apoio à decisão clínica.
- Não use markdown, negrito, asteriscos ou listas.
"""

    return _generate_interpretation(prompt, model_name, probability, risk_level)


def generate_image_interpretation(result: dict, model_name: str, threshold: float) -> str:
    probability = result["probability"]
    disease_detected = result["disease_detected"]
    risk_level = classify_risk(probability)

    prompt = f"""
Você é uma LLM integrada a um sistema de apoio ao diagnóstico médico.

Explique o resultado abaixo de forma clara, objetiva e responsável.

Dados do modelo:
- Modelo utilizado: {model_name}
- Tipo de exame: análise de imagem de ressonância por visão computacional
- Probabilidade estimada: {probability * 100:.2f}%
- Limiar de decisão: {threshold * 100:.2f}%
- Achado suspeito detectado pelo modelo: {"sim" if disease_detected else "não"}
- Nível de risco estimado: {risk_level}

Responda obrigatoriamente neste formato:

Resumo:
Risco:
Justificativa:
Recomendação:
Observação:

Regras:
- Não diga que o paciente tem diagnóstico confirmado.
- Não substitua avaliação médica nem laudo radiológico.
- Use linguagem adequada para profissionais da saúde.
- Seja objetivo.
- A recomendação deve ser compatível com o nível de risco.
- A observação deve deixar claro que o resultado é apoio à decisão clínica, baseado apenas na análise automática da imagem.
- Não use markdown, negrito, asteriscos ou listas.
"""

    return _generate_interpretation(prompt, model_name, probability, risk_level)
