import wfdb

annotations = wfdb.rdann("mit-bih\\102", "atr")

with open('testingScripts/annotations.txt', 'w') as ann:
    ann.write(str(annotations.aux_note))