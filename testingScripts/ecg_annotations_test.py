import wfdb

annotations = wfdb.rdann("mit-bih\\100", "atr")

print(annotations.symbol)