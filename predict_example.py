from metadata_bn import MetadataBN

m = MetadataBN.load("model.pkl")

# Edit this: put in whatever you know about a study. Field names must match your studies.csv columns.
record = {"organism": "Homo sapiens", "is_cell_line": "no", "disease_area": "cancer"}

print("Known:", record, "\n")
for field, top in m.predict(record, top_k=3).items():
    print(f"{field:18s}", ", ".join(f"{v} ({p:.2f})" for v, p in top))
