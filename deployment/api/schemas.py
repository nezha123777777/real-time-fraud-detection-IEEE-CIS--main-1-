"""Pydantic schemas for API requests and responses."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class FraudPredictionRequest(BaseModel):
    TransactionAmt: float = Field(..., description="Dollar amount of the transaction")
    TransactionDT: int = Field(..., description="Timestamp delta from reference")

    ProductCD: Optional[str] = Field(None, description="Product code")
    card1: Optional[int] = Field(None, description="Card identifier 1")
    card4: Optional[str] = Field(None, description="Card network")
    card6: Optional[str] = Field(None, description="Debit or credit")
    addr1: Optional[float] = Field(None, description="Billing region")
    addr2: Optional[float] = Field(None, description="Billing country")
    P_emaildomain: Optional[str] = Field(None, description="Purchaser email domain")
    R_emaildomain: Optional[str] = Field(None, description="Recipient email domain")

    D1: Optional[float] = None
    D2: Optional[float] = None
    D3: Optional[float] = None
    D4: Optional[float] = None
    D15: Optional[float] = None

    id_31: Optional[str] = Field(None, description="Browser identifier")
    id_33: Optional[str] = Field(None, description="Screen resolution")
    DeviceType: Optional[str] = Field(None, description="desktop or mobile")
    DeviceInfo: Optional[str] = Field(None, description="device information")

    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "TransactionAmt": 150.50,
                "TransactionDT": 86450,
                "ProductCD": "W",
                "card1": 10409,
                "card4": "visa",
                "card6": "debit",
                "addr1": 299.0,
                "D1": 14.0,
                "id_31": "chrome 70.0",
                "id_33": "1920x1080",
                "DeviceType": "desktop",
            }
        },
    }


class FraudPredictionResponse(BaseModel):
    TransactionID: str
    fraud_probability: float
    fraud_label: int = Field(..., description="0 = legit, 1 = fraud")
    action: str = Field(..., description="ALLOW, REVIEW, or BLOCK")
    execution_time_ms: float
    threshold: float
    review_threshold: float
    anomaly_flag: bool = False
    anomaly_score: Optional[float] = None
    model_version: str
    model_config = {"protected_namespaces": ()}


class FeatureContribution(BaseModel):
    feature: str
    feature_value: Optional[Any] = None
    shap_value: float
    abs_shap_value: float
    model_config = {"protected_namespaces": ()}


class FraudExplainResponse(FraudPredictionResponse):
    base_value: float
    feature_count: int
    top_features: list[FeatureContribution]
