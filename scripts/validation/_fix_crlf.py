"""Fix CRLF to LF in research diary."""
p = r"research_diary/04_tareas_post_reunion_diego.md"
with open(p, "rb") as f:
    data = f.read()
data = data.replace(b"\r\n", b"\n")
with open(p, "wb") as f:
    f.write(data)
print("Fixed CRLF -> LF")
