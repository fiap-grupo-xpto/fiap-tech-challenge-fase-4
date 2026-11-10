"""Streamlit view for the temporal monitoring API."""
import json
import time
import uuid

import pandas as pd
import requests
import streamlit as st


def render_anomalies(api_url):
    st.title("Monitoramento de anomalias — Fase 4")
    st.caption("Sinais vitais, prescrições e atividade durante a internação")
    st.info("Demonstração experimental com dados sintéticos. Os alertas indicam mudanças para revisão humana, não diagnósticos.")

    def api(method, path, **kwargs):
        response = requests.request(method, f"{api_url}/anomalies{path}", timeout=30, **kwargs)
        if not response.ok:
            raise ValueError(f"API {response.status_code}: {response.text[:1000]}")
        return response.json()

    if "anomaly_patient" not in st.session_state:
        st.session_state.anomaly_patient = "DEMO-" + uuid.uuid4().hex[:8]
    patient = st.text_input("Identificador do paciente", key="anomaly_patient")
    st.caption("Use um identificador novo para cada demonstração. Reenviar os mesmos eventos não duplica os alertas.")
    try:
        demo = api("GET", "/demo", params={"patient_id": patient})
        st.download_button("Baixar exemplo JSON sintético", json.dumps({"observations": demo["observations"]}, indent=2),
                           "monitoramento_sintetico.json", "application/json")
        if st.button("Reproduzir demonstração sintética", type="primary"):
            progress = st.progress(0, text="Reproduzindo histórico em ordem temporal…")
            notice = st.empty()
            for index, event in enumerate(demo["observations"]):
                result = api("POST", "/observations", json={"observations": [event]})
                for alert in result["new_alerts"]:
                    notice.warning(f"Alerta automático: {alert['message']} Horário: {alert['timestamp']}")
                progress.progress((index + 1) / len(demo["observations"]), text=f"Evento {index + 1}: {event['timestamp']}")
                time.sleep(0.1)
            st.success("Reprodução concluída. Consulte os gráficos e os alertas abaixo.")
        upload = st.file_uploader("Importar histórico JSON (até 500 eventos por envio)", type=["json"])
        if upload and st.button("Processar histórico"):
            if upload.size > 2 * 1024 * 1024:
                raise ValueError("Arquivo excede 2 MB.")
            batch = json.loads(upload.getvalue())
            if not isinstance(batch, dict) or not isinstance(batch.get("observations"), list):
                raise ValueError("Use o formato do exemplo JSON.")
            if any(not isinstance(e, dict) or e.get("patient_id") != patient for e in batch["observations"]):
                raise ValueError("Todos os eventos devem pertencer ao paciente selecionado.")
            result = api("POST", "/observations", json=batch)
            st.success(f"Histórico processado: {len(result['new_alerts'])} novo(s) alerta(s).")
        live = st.toggle("Atualização automática (a cada 5 s)", value=True, key="anomaly_live")
        st.button("Atualizar painel")
    except (requests.RequestException, ValueError) as exc:
        st.error(f"Não foi possível concluir a operação: {exc}")
        return

    # Only this fragment reruns on the timer, so alerts sent by other sources appear without a page reload.
    @st.fragment(run_every=5 if live else None)
    def board():
        try:
            render_board(api, patient)
        except (requests.RequestException, ValueError) as exc:
            st.error(f"Não foi possível atualizar o painel: {exc}")

    board()


def render_board(api, patient):
    data = api("GET", f"/patients/{patient}")
    observations, alerts = data["observations"], data["alerts"]
    # Toast only alerts that arrive after the panel was opened for this patient.
    known = st.session_state.setdefault("anomaly_known_alerts", {})
    if patient in known:
        for alert in alerts:
            if alert["id"] not in known[patient] and not alert["acknowledged_at"]:
                st.toast(f"Novo alerta: {alert['patient_id']} — {alert['code']}", icon="🚨")
    known[patient] = {alert["id"] for alert in alerts}
    left, middle, right = st.columns(3)
    left.metric("Observações", len(observations))
    middle.metric("Alertas registrados", len(alerts))
    right.metric("Não vistos", sum(a["acknowledged_at"] is None for a in alerts))
    st.caption("Painel exibe os últimos 500 registros e alertas. Horários em UTC. A reprodução é acelerada; não representa sensores conectados.")
    for kind, title in (("vitals", "Sinais vitais"), ("movement", "Atividade por janela de 5 minutos")):
        rows = [{"timestamp": e["timestamp"], **e["metrics"]} for e in observations if e["kind"] == kind]
        if rows:
            st.subheader(title)
            frame = pd.DataFrame(rows).set_index("timestamp")
            for metric in frame.columns:
                st.caption(metric)
                st.line_chart(frame[[metric]])
    prescriptions = [{"timestamp": e["timestamp"], **e["prescription"]} for e in observations if e["kind"] == "prescription"]
    if prescriptions:
        st.subheader("Evolução das prescrições")
        st.dataframe(prescriptions, use_container_width=True)
    st.subheader("Alertas para revisão")
    if not alerts:
        st.info("Nenhum alerta registrado. Histórico insuficiente ou ausência de alerta não comprova normalidade clínica.")
    for alert in alerts:
        with st.expander(f"{'Visto' if alert['acknowledged_at'] else 'Novo'} | {alert['timestamp']} | {alert['code']}"):
            st.warning(alert["message"])
            st.json(alert["evidence"])
            delivery = alert.get("delivery")
            if delivery and delivery["status"] in ("pending", "sending"):
                label = "Envio pendente" if delivery["status"] == "pending" else "Envio em andamento"
                st.caption(f"{label} — tentativas iniciadas: {delivery['attempts']}")
            for notice in alert.get("notifications", []):
                label = {"sent": "Enviado à equipe", "failed": "Falha no envio à equipe",
                         "panel_only": "Somente no painel"}.get(notice["status"], notice["status"])
                st.caption(f"{label} em {notice['at']} — {notice['detail']}")
            if alert["acknowledged_at"]:
                st.caption(f"Marcado como visto em {alert['acknowledged_at']}")
            elif st.button("Marcar como visto", key=alert["id"]):
                api("POST", f"/alerts/{alert['id']}/acknowledge")
                st.rerun()
    st.caption("Marcar como visto é um registro demonstrativo; não representa aprovação médica autenticada.")
    with st.expander("Histórico e estado do detector"):
        st.dataframe([{"timestamp": e["timestamp"], "kind": e["kind"], "state": e["analysis"]["state"],
                       "event_id": e["event_id"]} for e in observations], use_container_width=True)
    st.download_button("Baixar relatório do paciente", json.dumps(data, ensure_ascii=False, indent=2),
                       "relatorio_anomalias.json", "application/json")
