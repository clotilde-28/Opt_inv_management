# pip install tardis-dev
# requires Python >=3.9
from tardis_dev import download_datasets

download_datasets(
    exchange="binance",
    data_types=[
        "book_ticker",
        "trades"
    ],
    from_date="2024-04-01",
    to_date="2024-04-02",
    symbols=["BTCUSDT"],
    download_dir="./data"
)