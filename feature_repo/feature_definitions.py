from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int32, Int64, String, UnixTimestamp

# Define Entities
transaction = Entity(
    name="transaction",
    join_keys=["TransactionID"],
    description="A unique transaction identifier",
)

card = Entity(
    name="card",
    join_keys=["card1"],
    description="The primary card identifier (card1)",
)

# Define Data Source
# Pointing to the cleaned dataset that now has the event_timestamp
transaction_source = FileSource(
    path="c:/Users/User/Documents/pfa/ieee_fraud_project/data/interim/feature_ready_train.parquet",
    event_timestamp_column="event_timestamp",
)

# Feature View for Transaction-level features
transaction_stats_fv = FeatureView(
    name="transaction_stats",
    entities=[transaction],
    ttl=timedelta(days=1),
    schema=[
        Field(name="TransactionAmt", dtype=Float32),
        Field(name="log_TransactionAmt", dtype=Float32),
        Field(name="amt_decimal", dtype=Float32),
        Field(name="amt_is_round", dtype=Int32),
        Field(name="hour", dtype=Int32),
        Field(name="day_of_week", dtype=Int32),
        Field(name="is_weekend", dtype=Int32),
        Field(name="is_night", dtype=Int32),
    ],
    online=True,
    source=transaction_source,
)

# Feature View for Card-level ID info (example of another view)
card_info_fv = FeatureView(
    name="card_info",
    entities=[card],
    ttl=timedelta(days=365),
    schema=[
        Field(name="card2", dtype=Float32),
        Field(name="card3", dtype=Float32),
        Field(name="card5", dtype=Float32),
        Field(name="addr1", dtype=Float32),
    ],
    online=True,
    source=transaction_source,
)
