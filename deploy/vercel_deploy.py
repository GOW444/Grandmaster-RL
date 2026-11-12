"""Deploy frontend/dist + vercel.json to Vercel via REST API (static, prebuilt). Never prints the token."""

import base64
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(r"d:\projects\GrandMasterRL\repo")
TOKEN = [l.split("=", 1)[1].strip().strip("\"'") for l in (REPO / "deploy.env").read_text(encoding="utf-8").splitlines()
         if l.startswith("VERCEL_TOKEN=")][0]
NAME = "grandmaster-rl"


def call(method, path, body=None):
    req = urllib.request.Request(
        "https://api.vercel.com" + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:1000]


def main():
    dist = REPO / "frontend" / "dist"
    files = []
    for p in sorted(dist.rglob("*")):
        if p.is_file():
            files.append({"file": p.relative_to(dist).as_posix(), "data": base64.b64encode(p.read_bytes()).decode(),
                          "encoding": "base64"})
    vj = (REPO / "frontend" / "vercel.json").read_bytes()
    files.append({"file": "vercel.json", "data": base64.b64encode(vj).decode(), "encoding": "base64"})
    print("uploading", len(files), "files,", sum(len(f["data"]) for f in files) // 1024, "KB (b64)")

    body = {
        "name": NAME,
        "target": "production",
        "files": files,
        "projectSettings": {"framework": None, "buildCommand": None, "installCommand": None,
                            "outputDirectory": None, "devCommand": None},
    }
    st, d = call("POST", "/v13/deployments?skipAutoDetectionConfirmation=1", body)
    if st not in (200, 201):
        sys.exit(f"deploy HTTP {st}: {d}")
    did = d["id"]
    print("deployment", did, d.get("readyState"))
    for _ in range(60):
        st, d = call("GET", f"/v13/deployments/{did}")
        state = d.get("readyState")
        if state in ("READY", "ERROR", "CANCELED"):
            break
        time.sleep(3)
    print("state:", state)
    print("deployment url: https://" + d.get("url", ""))
    print("aliases:", ["https://" + a for a in d.get("alias", [])])
    # Make sure production deployments are publicly reachable (no Vercel login wall).
    st, proj = call("GET", f"/v9/projects/{NAME}")
    if st == 200:
        print("project:", proj.get("id"), "| ssoProtection:", proj.get("ssoProtection"))


if __name__ == "__main__":
    main()
