from src.model_logic import load_runtime_artifacts
from dataclasses import asdict, is_dataclass

art = load_runtime_artifacts("artifacts")

print("type:", type(art))

if is_dataclass(art):
    d = asdict(art)
    print("fields:", list(d.keys()))
elif isinstance(art, dict):
    d = art
    print("keys:", list(d.keys()))
else:
    d = {k: getattr(art, k) for k in dir(art) if not k.startswith("_")}
    print("attributes:", list(d.keys()))

matrix = d.get("matrix")
coords = d.get("coords")
pca_meta = d.get("pca_meta")

print("matrix shape:", getattr(matrix, "shape", None))
print("coords shape:", getattr(coords, "shape", None))
print("coords columns:", list(getattr(coords, "columns", [])) if coords is not None else None)
print("pca_meta keys:", list(pca_meta.keys()) if hasattr(pca_meta, "keys") else None)
print("clusters present?", d.get("clusters") is not None)
print("themes present?", d.get("themes") is not None)