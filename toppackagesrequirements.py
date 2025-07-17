import json
import subprocess


def get_top_level_conda_deps(env_name=None):
    cmd = ["conda", "env", "export", "--json"]
    if env_name:
        cmd += ["-n", env_name]
    result = subprocess.run(cmd, capture_output=True, text=True)
    env_data = json.loads(result.stdout)
    top_deps = env_data.get("dependencies", [])

    # Filter out pip section and subdependencies
    pip_deps = []
    if isinstance(top_deps, list):
        deps = []
        for dep in top_deps:
            if isinstance(dep, dict) and "pip" in dep:
                pip_deps = dep["pip"]
            elif isinstance(dep, str):
                name_ver = dep.split("=")
                if len(name_ver) >= 2:
                    deps.append(f"{name_ver[0]}=={name_ver[1]}")
        return deps + pip_deps
    return []


deps = get_top_level_conda_deps()
with open("requirements.txt", "w") as f:
    for dep in deps:
        f.write(dep + "\n")
