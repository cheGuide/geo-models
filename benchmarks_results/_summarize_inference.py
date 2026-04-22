import json
from pathlib import Path

p = Path(__file__).with_name("custom_images_inference.json")
d = json.loads(p.read_text(encoding="utf-8"))


def short_name(k: str) -> str:
    if len(k) > 50:
        return "…" + k[-47:]
    return k


print("=== Сводка: GeoCLIP top-1 (baseline / fine-tuned), Kopernik (lat, lon), Revisit VPR (pred с БД, similarity) ===\n")
keys = sorted(
    set(d.get("geoclip", {}).get("baseline", {}))
    | set(d.get("geoclip", {}).get("fine_tuned", {}))
)
for k in keys:
    if k.startswith("_"):
        continue
    gb = d["geoclip"]["baseline"].get(k, {})
    gf = d["geoclip"].get("fine_tuned", {}).get(k, {})
    kop = d.get("kopernik_finetuned", {}).get(k, {})
    rv = d.get("revisit_vpr_finetuned", {}).get(k, {})
    t1b = (gb.get("top3") or [{}])[0]
    t1f = (gf.get("top3") if isinstance(gf, dict) else None) or [{}]
    t1f = t1f[0] if t1f else {}
    print(short_name(k))
    if t1b:
        print(f"  GeoCLIP baseline  top1: {t1b.get('lat', 0):.4f}, {t1b.get('lon', 0):.4f}  p={t1b.get('prob', 0):.4f}")
    else:
        print(f"  GeoCLIP baseline: {gb.get('error', '—')}")
    if t1f:
        print(f"  GeoCLIP fine-tuned top1: {t1f.get('lat', 0):.4f}, {t1f.get('lon', 0):.4f}  p={t1f.get('prob', 0):.4f}")
    elif isinstance(gf, dict):
        print(f"  GeoCLIP fine-tuned: {gf.get('error', '—')}")
    if "lat" in kop:
        print(f"  Kopernik:  {kop['lat']:.5f}, {kop['lon']:.5f}")
    else:
        print(f"  Kopernik: {kop.get('error', '—')}")
    if "pred_lat" in rv:
        print(
            f"  Revisit:   {rv['pred_lat']:.5f}, {rv['pred_lon']:.5f}  sim={rv.get('similarity', 0):.4f}  file={rv.get('matched_filename', '')[:50]}"
        )
    else:
        print(f"  Revisit: {rv.get('error', '—')}")
    print()
