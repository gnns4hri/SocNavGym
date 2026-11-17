#!/usr/bin/env python3
"""Removes the old keys `vision: viewer: width` and `vision: viewer: height` from YAML configurations.

The 3D view now has the size given by `rendering: resolution_x, resolution_y`, and a configuration that still has the old keys
is refused when it is loaded. This script takes the two keys out of every YAML file that has them and leaves the rest of the
file as it is (comments, order, spacing): it removes lines, it does not write the file again from its parsed content.

    python remove_viewer_size.py [FILE or FOLDER ...]      default: the current folder (folders are searched recursively)
    python remove_viewer_size.py --dry-run FOLDER          only says what it would do

A file is changed only if, read again, it is the same configuration without the two keys; otherwise it is left alone and named
(`NOT CHANGED`). Folders of other software are not searched (.git, virtual environments, node_modules, site-packages).
Other `width` / `height` keys (a camera's, for instance) are not touched."""
import os, re, sys, copy
import yaml

KEYS = ("width", "height")
SKIP = {".git", "node_modules", "site-packages", "__pycache__"}


def indent(line): return len(line) - len(line.lstrip(" "))
def is_content(line): return bool(line.strip()) and not line.lstrip().startswith("#")


def flow_without_keys(text):
    """`{a: 1, width: 2, b: {c: 3}}` -> `{a: 1, b: {c: 3}}` (the entries `width` and `height` of the outer mapping removed)."""
    a, b = text.index("{"), text.rindex("}"); inner = text[a + 1:b]; parts = []; depth = 0; cur = ""
    for ch in inner:
        if ch in "{[": depth += 1
        elif ch in "}]": depth -= 1
        if ch == "," and depth == 0: parts.append(cur); cur = ""
        else: cur += ch
    parts.append(cur)
    kept = [p.strip() for p in parts if p.strip() and p.split(":")[0].strip().strip("'\"") not in KEYS]
    return text[:a] + "{" + ", ".join(kept) + "}" + text[b + 1:]


def edit(text):
    """The text without the two keys of vision.viewer, and the names of the keys removed."""
    lines = text.splitlines(keepends=True); out = []; removed = []
    i = next((k for k, l in enumerate(lines) if re.match(r"vision\s*:", l)), None)                     # the top-level `vision:` section
    if i is None: return text, removed
    end = next((k for k in range(i + 1, len(lines)) if is_content(lines[k]) and indent(lines[k]) == 0), len(lines))
    child = min((indent(l) for l in lines[i + 1:end] if is_content(l)), default=None)                    # indentation of the section's own keys
    v = next((k for k in range(i + 1, end) if is_content(lines[k]) and indent(lines[k]) == child and re.match(r"\s*viewer\s*:", lines[k])), None)
    if v is None: return text, removed
    rest = lines[v].split(":", 1)[1]
    if rest.lstrip().startswith("{"):                                                                    # viewer: {enabled: true, width: 1280}
        head = lines[v][:len(lines[v]) - len(rest)]; new = head + flow_without_keys(rest)
        removed = [k for k in KEYS if re.search(r"[{,]\s*['\"]?%s['\"]?\s*:" % k, rest)]
        lines[v] = new; return "".join(lines), removed
    vend = next((k for k in range(v + 1, end) if is_content(lines[k]) and indent(lines[k]) <= child), end)   # the lines of `viewer:`
    sub = min((indent(l) for l in lines[v + 1:vend] if is_content(l)), default=None)
    for k, l in enumerate(lines):
        m = re.match(r"\s*['\"]?(\w+)['\"]?\s*:", l) if (v < k < vend and is_content(l) and indent(l) == sub) else None
        if m and m.group(1) in KEYS: removed.append(m.group(1))
        else: out.append(l)
    return "".join(out), removed


def one(path, dry):
    try:
        with open(path) as f: text = f.read()
        before = yaml.safe_load(text)
    except Exception as e:
        return None if not text_mentions(path) else f"NOT CHANGED {path}: cannot be read as YAML ({type(e).__name__})"
    viewer = before.get("vision", {}).get("viewer") if isinstance(before, dict) and isinstance(before.get("vision"), dict) else None
    if not isinstance(viewer, dict) or not any(k in viewer for k in KEYS): return None
    want = copy.deepcopy(before)
    for k in KEYS: want["vision"]["viewer"].pop(k, None)
    new, removed = edit(text)
    try: after = yaml.safe_load(new)
    except Exception: after = None
    if isinstance(after, dict) and isinstance(after.get("vision"), dict) and after["vision"].get("viewer") is None and want["vision"]["viewer"] == {}:
        after["vision"]["viewer"] = {}                                   # `viewer:` left with no keys reads as an empty value: the same thing
    if after != want: return f"NOT CHANGED {path}: has the old keys, but this script cannot remove them safely (edit it by hand)"
    if not dry:
        tmp = path + ".tmp"
        with open(tmp, "w") as f: f.write(new)
        os.chmod(tmp, os.stat(path).st_mode & 0o7777); os.replace(tmp, path)
    return f"{'would remove' if dry else 'removed'} viewer.{' and viewer.'.join(removed)}: {path}"


def text_mentions(path):
    try:
        with open(path, errors="replace") as f: t = f.read()
        return bool(re.search(r"^vision\s*:", t, re.M) and re.search(r"viewer\s*:", t))
    except Exception: return False


def files(paths):
    for p in paths:
        if os.path.isfile(p): yield p
        else:
            for d, subs, names in os.walk(p):
                subs[:] = sorted(s for s in subs if s not in SKIP and not os.path.exists(os.path.join(d, s, "pyvenv.cfg")))
                for n in sorted(names):
                    if n.endswith((".yaml", ".yml")): yield os.path.join(d, n)


def main(argv):
    if "-h" in argv or "--help" in argv: print(__doc__); return 0
    dry = "--dry-run" in argv; paths = [a for a in argv if not a.startswith("--")] or ["."]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing: print("not found:", ", ".join(missing)); return 2
    n = changed = failed = 0
    for f in files(paths):
        n += 1; r = one(f, dry)
        if r: print(r); changed += not r.startswith("NOT CHANGED"); failed += r.startswith("NOT CHANGED")
    print(f"{n} YAML files looked at; {changed} {'to change' if dry else 'changed'}; {failed} not changed (named above)")
    return 1 if failed else 0


if __name__ == "__main__": sys.exit(main(sys.argv[1:]))
