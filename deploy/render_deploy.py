"""Create/inspect the Render web service via the REST API. Never prints secrets."""

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENV = {}
for line in (REPO / "deploy.env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        ENV[k.strip()] = v.strip().strip('"').strip("'")

API = "https://api.render.com/v1"
KEY = ENV["RENDER_API_KEY"]
NAME = "grandmaster-rl-api"


def call(method, path, body=None):
    req = urllib.request.Request(
        API + path,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {KEY}", "Accept": "application/json", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            txt = r.read().decode()
            return r.status, json.loads(txt) if txt else None
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:800]


def find_service():
    st, data = call("GET", f"/services?name={NAME}&limit=20")
    for item in data or []:
        s = item.get("service", item)
        if s.get("name") == NAME:
            return s
    return None


def create():
    st, owners = call("GET", "/owners?limit=20")
    if st != 200:
        sys.exit(f"owners: HTTP {st} {owners}")
    owner = owners[0]["owner"]
    print("workspace:", owner.get("name"), owner.get("type"))
    if find_service():
        print("service already exists")
        return
    body = {
        "type": "web_service",
        "name": NAME,
        "ownerId": owner["id"],
        "repo": "https://github.com/GOW444/Grandmaster-RL",
        "branch": "main",
        "rootDir": "backend",
        "autoDeploy": "yes",
        "envVars": [
            {"key": "DATABASE_URL", "value": ENV["NEON_DATABASE_URL"]},
            {"key": "JWT_SECRET", "generateValue": True},
            {"key": "PYTHON_VERSION", "value": "3.12.7"},
            {"key": "CORS_ORIGINS", "value": '["https://grandmaster-rl.vercel.app"]'},
        ],
        "serviceDetails": {
            "runtime": "python",
            "plan": "free",
            "region": "singapore",
            "healthCheckPath": "/api/health",
            "envSpecificDetails": {
                "buildCommand": "pip install -r requirements.txt",
                "startCommand": "uvicorn app.main:app --host 0.0.0.0 --port $PORT",
            },
        },
    }
    st, data = call("POST", "/services", body)
    if st not in (200, 201):
        sys.exit(f"create: HTTP {st} {data}")
    svc = data.get("service", data)
    print("created:", svc.get("id"), svc.get("serviceDetails", {}).get("url"))


def status():
    s = find_service()
    if not s:
        sys.exit("no service")
    st, deploys = call("GET", f"/services/{s['id']}/deploys?limit=1")
    d = deploys[0]["deploy"] if deploys else {}
    print("url:", s.get("serviceDetails", {}).get("url"), "| deploy:", d.get("id"), d.get("status"))


def logs():
    s = find_service()
    st, owners = call("GET", "/owners?limit=1")
    oid = owners[0]["owner"]["id"]
    st, data = call("GET", f"/logs?ownerId={oid}&resource={s['id']}&limit=60&direction=backward")
    if st != 200:
        print("logs HTTP", st, data)
        return
    for l in reversed(data.get("logs", [])):
        msg = l.get("message", "")
        if "neon.tech" in msg or "postgresql://" in msg:
            msg = "<redacted>"
        print(msg[:220])


def redeploy():
    s = find_service()
    st, data = call("POST", f"/services/{s['id']}/deploys", {"clearCache": "do_not_clear"})
    print("redeploy:", st, (data or {}).get("id") if isinstance(data, dict) else data)


def set_env(key, value):
    s = find_service()
    st, data = call("PUT", f"/services/{s['id']}/env-vars/{key}", {"value": value})
    print("set", key, "HTTP", st)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "set_env":
        set_env(sys.argv[2], sys.argv[3])
    else:
        {"create": create, "status": status, "logs": logs, "redeploy": redeploy}[cmd]()
