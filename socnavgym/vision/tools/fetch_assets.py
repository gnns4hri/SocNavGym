"""Downloads the source files of the 3D assets into <repository>/vision_cache/ (not part of the repository).

    python3 socnavgym/vision/tools/fetch_assets.py

Sources (both compatible with SocNavGym's GPL v3):
  - furniture: Poly Haven (https://polyhaven.com), CC0: metal_office_desk, SchoolChair_01, potted_plant_02, classic_laptop
  - surfaces:  Poly Haven textures, CC0: laminate_floor_02 (floor), white_plaster_02 (walls)
  - people:    Microsoft Rocketbox avatar library (https://github.com/microsoft/Microsoft-Rocketbox), MIT licence:
               Male_Adult_01, Female_Adult_01 and their neutral walking / standing animations
The assets that the simulator loads (socnavgym/vision/assets/*.blend, *.npz) are made from these by build_assets.py.
Only the standard library is used."""
import json, os, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CACHE = os.environ.get("SOCNAV_VISION_CACHE", os.path.join(REPO, "vision_cache"))
UA = {"User-Agent": "socnavgym-vision-asset-fetch/0.1"}
POLYHAVEN = ["metal_office_desk", "SchoolChair_01", "potted_plant_02", "classic_laptop"]
TEXTURES = ["laminate_floor_02", "white_plaster_02"]; MAPS = ["Diffuse", "nor_gl", "Rough"]
RB = "https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/"
AVATARS = {"Male_Adult_01": "m002", "Female_Adult_01": "f001"}
ANIMS = [("xy", "m_walk_neutral"), ("xy", "f_walk_neutral"), ("static", "m_idle_neutral_01"), ("static", "f_idle_neutral_01")]
TEX = ["body_color", "body_normal", "body_specular", "head_color", "head_normal", "head_specular", "opacity_color"]


def get(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0: return
    os.makedirs(os.path.dirname(path), exist_ok=True); print("  ", url.split("/")[-1], flush=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180) as r, open(path + ".part", "wb") as f: f.write(r.read())
    os.replace(path + ".part", path)


def main():
    for a in POLYHAVEN:
        with urllib.request.urlopen(urllib.request.Request(f"https://api.polyhaven.com/files/{a}", headers=UA), timeout=60) as r: g = json.load(r)["gltf"]["1k"]["gltf"]
        d = os.path.join(CACHE, "polyhaven", a); get(g["url"], os.path.join(d, os.path.basename(g["url"])))
        for rel, inc in g.get("include", {}).items(): get(inc["url"], os.path.join(d, rel))
    for a in TEXTURES:
        with urllib.request.urlopen(urllib.request.Request(f"https://api.polyhaven.com/files/{a}", headers=UA), timeout=60) as r: g = json.load(r)
        for k in MAPS: u = g[k]["1k"]["jpg"]["url"]; get(u, os.path.join(CACHE, "polyhaven", a, k + ".jpg"))
    d = os.path.join(CACHE, "rocketbox")
    for name, code in AVATARS.items():
        get(f"{RB}Avatars/Adults/{name}/Export/{name}.fbx", os.path.join(d, "Avatars", "Adults", name, "Export", f"{name}.fbx"))
        for t in TEX: get(f"{RB}Avatars/Adults/{name}/Textures/{code}_{t}.tga", os.path.join(d, "Avatars", "Adults", name, "Textures", f"{code}_{t}.tga"))
    for k, n in ANIMS: get(f"{RB}Animations/all_animations_max_motextr_{k}/{n}.max.fbx", os.path.join(d, "Animations", f"all_animations_max_motextr_{k}", f"{n}.max.fbx"))
    print("sources in", CACHE)


if __name__ == "__main__": main()
