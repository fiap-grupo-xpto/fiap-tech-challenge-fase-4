"""Streamlit form and review report for a single rehabilitation session."""

import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import requests
import streamlit as st


def render_multimodal(api_url, exercises):
    st.title("Caso multimodal")
    mode = st.radio("Origem dos dados", ["Demonstração sintética", "Enviar áudio e vídeo"], horizontal=True)
    previous = st.session_state.get("multimodal_report")
    if mode == "Demonstração sintética":
        st.info("A nota e os resultados de áudio e vídeo são simulados. Os alertas passam pelo detector e pela fila de envio.")
        if "multimodal_demo_id" not in st.session_state:
            st.session_state.multimodal_demo_id = f"DEMO-{uuid4().hex[:8]}"
        patient = st.text_input("Identificador do paciente de demonstração", key="multimodal_demo_id")
        if st.button("Executar caso sintético", type="primary", disabled=not patient.strip()):
            _submit(api_url, "/demo", params={"patient_id": patient.strip()})
    else:
        _upload_form(api_url, exercises, previous)

    report = st.session_state.get("multimodal_report")
    selected_mode = "demo" if mode == "Demonstração sintética" else "upload"
    if report and report["mode"] == selected_mode:
        _render_report(report)


def _upload_form(api_url, exercises, previous):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    patient_default = previous["patient_id"] if previous else ""
    window = previous["window"] if previous else {
        "start": (now - timedelta(hours=1)).isoformat(),
        "end": now.isoformat(),
    }
    with st.form("multimodal_upload"):
        patient = st.text_input(
            "Identificador no painel de anomalias", value=patient_default, key="multimodal_patient"
        )
        start_col, end_col = st.columns(2)
        start = start_col.text_input(
            "Início da sessão (ISO 8601 com fuso)", value=window["start"], key="multimodal_start"
        )
        end = end_col.text_input(
            "Fim da sessão (ISO 8601 com fuso)", value=window["end"], key="multimodal_end"
        )
        clinical_note = st.text_area("Nota da sessão", max_chars=4000, key="multimodal_note")
        exercise = st.selectbox("Exercício", list(exercises), format_func=exercises.get)
        audio = st.file_uploader("Áudio (WAV PCM de 16 bits, 16 kHz, mono, até 60 s)", type=["wav"])
        video = st.file_uploader("Vídeo de uma repetição (MP4)", type=["mp4"])
        submit = st.form_submit_button("Analisar sessão", type="primary")
    if submit:
        if not (patient.strip() and clinical_note.strip() and audio and video):
            st.error("Informe o paciente, a nota e os dois arquivos da sessão.")
            return
        _submit(
            api_url, "",
            data={"patient_id": patient.strip(), "start": start, "end": end,
                  "clinical_note": clinical_note, "exercise_id": exercise},
            files={"audio": (audio.name, audio.getvalue(), "audio/wav"),
                   "video": (video.name, video.getvalue(), "video/mp4")},
        )


def _submit(api_url, suffix, **kwargs):
    st.session_state.pop("multimodal_report", None)
    with st.spinner("Analisando o caso…"):
        try:
            response = requests.request("POST", f"{api_url}/analyze-rehabilitation-case{suffix}",
                                        timeout=900, **kwargs)
            if not response.ok:
                detail = response.json().get("detail") if response.headers.get(
                    "content-type", ""
                ).startswith("application/json") else None
                st.error(f"A análise retornou HTTP {response.status_code}. {detail or ''}")
                return
            report = response.json()
        except (requests.RequestException, ValueError) as exc:
            st.error(f"Falha na análise do caso. {exc}")
            return
    st.session_state.multimodal_report = report


def _render_report(report):
    levels = {
        "prioritized_review": "Revisão prioritária",
        "review": "Revisão indicada",
        "no_signal": "Sem sinal pelos critérios experimentais",
    }
    review = report["review"]
    window = report["window"]
    st.header(levels[review["level"]])
    st.caption(f"Paciente {report['patient_id']} | {window['start']} a {window['end']}")
    st.write(review["reason"])
    st.caption("Resultado experimental para revisão humana.")

    st.subheader("Evidências por modalidade")
    with st.expander("Texto da sessão", expanded=True):
        st.text(report["text"]["clinical_note"])
        _show_signals(report["signals"]["text"])
    with st.expander("Áudio e transcrição", expanded=True):
        audio = report["audio"]
        st.text(audio["transcript"])
        st.caption(f"Sentimento: {audio['sentiment']['label']}")
        if audio["azure_critical_terms"]:
            st.dataframe(audio["azure_critical_terms"], hide_index=True)
        _show_signals(report["signals"]["audio"])
    with st.expander("Vídeo e pose", expanded=True):
        st.write(report["video"]["deviation_report"]["summary"])
        _show_signals(report["signals"]["video"])

    st.subheader("Alertas confirmados na janela")
    if report["confirmed_alerts"]:
        st.dataframe([
            {"Horário": alert["timestamp"], "Alerta": alert["message"], "ID": alert["id"]}
            for alert in report["confirmed_alerts"]
        ], hide_index=True)
        st.caption("Consulte a entrega e marque o alerta como visto no painel de anomalias com este identificador.")
    else:
        st.info("Nenhum alerta confirmado no monitoramento para esta janela.")
    st.download_button("Baixar relatório do caso (JSON)", json.dumps(report, ensure_ascii=False, indent=2),
                       file_name="caso_multimodal.json", mime="application/json")


def _show_signals(signals):
    if signals:
        st.write("Sinais usados na revisão: " + ", ".join(signals))
