from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Prescription(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    medication: str = Field(min_length=1, max_length=100, pattern=r"\S")
    dose: float = Field(gt=0)
    unit: Literal["mg", "g", "mcg", "mL", "UI"]
    frequency_per_day: float = Field(gt=0, le=100)
    route: str = Field(min_length=1, max_length=40, pattern=r"\S")
    status: Literal["active", "stopped"] = "active"


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    event_id: str = Field(min_length=1, max_length=100, pattern=r"^[\w.-]+$")
    patient_id: str = Field(min_length=1, max_length=80, pattern=r"^[\w.-]+$")
    timestamp: datetime
    kind: Literal["vitals", "movement", "prescription"]
    context: Literal["awake", "rest"] = "awake"
    metrics: dict[str, float] = Field(default_factory=dict)
    prescription: Prescription | None = None

    @field_validator("timestamp")
    @classmethod
    def utc_timestamp(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp precisa incluir fuso horário")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_payload(self):
        ranges = {
            "heart_rate": (0, 350), "systolic": (0, 350),
            "diastolic": (0, 250), "spo2": (0, 100),
        }
        if self.kind == "prescription":
            if self.prescription is None or self.metrics:
                raise ValueError("prescription exige prescrição e metrics vazio")
        else:
            expected = set(ranges) if self.kind == "vitals" else {"activity_fraction"}
            if self.prescription is not None or set(self.metrics) != expected:
                raise ValueError(f"metrics deve conter exatamente {sorted(expected)}")
            for key, value in self.metrics.items():
                low, high = ranges.get(key, (0, 1))
                if not low <= value <= high:
                    raise ValueError(f"Valor inválido para {key}")
            if self.kind == "vitals" and self.metrics["diastolic"] >= self.metrics["systolic"]:
                raise ValueError("Pressão diastólica deve ser menor que sistólica")
        return self

    @property
    def stream(self):
        if self.prescription:
            return "prescription:" + self.prescription.medication.strip().casefold()
        return self.kind


class ObservationBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observations: list[Observation] = Field(min_length=1, max_length=500)
