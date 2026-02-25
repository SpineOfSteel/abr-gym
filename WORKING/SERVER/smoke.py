import urllib.request

try:
    with urllib.request.urlopen("http://localhost:8333/", timeout=0.15) as r:
        print("[MPC_HTTP] ping GET ok:", r.status)
except Exception as e:
    print("[MPC_HTTP] ping GET failed:", e)