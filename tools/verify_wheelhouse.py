"""Check that every wheel and sdist in a local wheel cache is byte-identical to the file the
official index publishes (D-029).

For each .whl or .tar.gz, the sha256 is compared with the hash published by:
  - pypi.org (JSON API) for packages hosted there;
  - pypi.nvidia.com (the simple index's #sha256 fragments) for NVIDIA-hosted files;
  - the #sha256 pinned in requirements.txt for direct-URL requirements (torch, torchvision).
Fails if any file's hash doesn't match a published one, no published hash is found, or the
folder holds anything else.

Usage: python3 tools/verify_wheelhouse.py WHEELHOUSE requirements.txt
"""

import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode()


def pypi_hashes(name, version):
    try:
        d = json.loads(get(f"https://pypi.org/pypi/{name}/{version}/json"))
        return {u["filename"]: u["digests"]["sha256"] for u in d["urls"]}
    except Exception:  # noqa: BLE001
        return {}


def nvidia_hashes(name):
    try:
        html = get(f"https://pypi.nvidia.com/{name}/")
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for href in re.findall(r'href="([^"]+)"', html):
        if "#sha256=" in href:
            url, h = href.split("#sha256=")
            out[urllib.request.unquote(url.rsplit("/", 1)[-1])] = h
    return out


def main():
    wh, req = Path(sys.argv[1]), Path(sys.argv[2])
    pinned = {}
    for line in req.read_text().splitlines():
        m = re.match(r"\s*([A-Za-z0-9_.-]+)\s*@\s*(\S+)#sha256=([0-9a-f]+)", line)
        if m:
            pinned[urllib.request.unquote(m.group(2).rsplit("/", 1)[-1])] = m.group(3)
    bad = 0
    wheels = sorted(list(wh.glob("*.whl")) + list(wh.glob("*.tar.gz")))
    for w in wheels:
        if w.name.endswith(".tar.gz"):
            name, version = w.name[: -len(".tar.gz")].rsplit("-", 1)
        else:
            name, version = w.name.split("-")[:2]
        name = re.sub(r"[-_.]+", "-", name).lower()
        got = sha256(w)
        if w.name in pinned:
            src, want = "requirements.txt", pinned[w.name]
        else:
            want = pypi_hashes(name, version).get(w.name)
            src = "pypi.org"
            if want is None:
                want, src = nvidia_hashes(name).get(w.name), "pypi.nvidia.com"
        ok = want is not None and got == want
        bad += not ok
        print(f"{'OK ' if ok else 'BAD'} {w.name}  ({src}{'' if want else ': no published hash'})")
    others = [p.name for p in wh.iterdir() if not p.name.endswith((".whl", ".tar.gz"))]
    print(f"{len(wheels)} files checked, {bad} not matching; non-wheel files: {others}")
    sys.exit(1 if bad or others else 0)


if __name__ == "__main__":
    main()
