import wfdb

annotations = wfdb.rdheader("training2017/training2017/A00038", "hea")

record = wfdb.rdrecord("training2017/training2017/A00038")


print(annotations)