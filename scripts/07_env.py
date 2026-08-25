"""Record hardware, OS, runtime and package versions for the Method section."""
import json, platform, subprocess, sys, os
def sh(c):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return None
env = {
  "os": f"{platform.system()} {platform.release()} {platform.version()}",
  "machine": platform.machine(),
  "cpu": sh('powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).Name"'),
  "cores_logical": os.cpu_count(),
  "ram_bytes": int(sh('powershell -NoProfile -Command "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"') or 0),
  "node": sh("node --version"), "npm": sh("npm --version"),
  "python": sys.version.split()[0], "cargo": sh("cargo --version"), "rustc": sh("rustc --version"),
  "packages": {}, "python_packages": {},
}
for p in ["@orama/orama","sqlite-vec","better-sqlite3","@lancedb/lancedb"]:
    v = sh(f'node -e "console.log(require(\'./node_modules/{p}/package.json\').version)"')
    env["packages"][p] = v
for m in ["numpy","onnxruntime","transformers","tokenizers"]:
    env["python_packages"][m] = sh(f'python -c "import {m};print({m}.__version__)"')
env["qdrant_edge_crate"] = sh("grep -m1 'qdrant-edge' engines-native/qe-napi/Cargo.toml")
env["corpus_commits"] = {}
for name, d in [("mdn","corpus-src"),("dotnet","corpus-dotnet"),("k8s","corpus-k8s"),("hass","corpus-ha")]:
    env["corpus_commits"][name] = sh(f"git -C {d} rev-parse HEAD")
env["model"] = {"name":"sentence-transformers/all-MiniLM-L6-v2","dim":384,
                "max_seq_length":256,"pooling":"mean","normalized":True,
                "runtime":"onnxruntime CPUExecutionProvider"}
json.dump(env, open("data/env.json","w"), indent=1)
print(json.dumps(env, indent=1))
