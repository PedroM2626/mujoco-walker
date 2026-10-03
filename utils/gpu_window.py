"""Measure the GPU window a long run is about to start in, and refuse a bad one.

Two documented losses motivated this. A Dreamer 1M attempt died at 84,456 steps with
`CUDA error: unspecified launch failure` inside a window where two other processes were also on
the device; and two runs of one identical build measured 15.6 and 68 env-steps/s, a 4.4x spread
the README attributes to desktop contexts queuing in the driver. Neither is fixable from inside
the trainer - the card is shared - but both are *attributable* if the window is recorded, and the
first is avoidable if a run says no before it spends eighteen hours.

The signal that works is `memory.used` before we allocate, not the process list: on this box five
desktop processes hold a CUDA context at all times (Medal, Overwolf, two Brave renderers, one the
driver will not name), and `--query-compute-apps` reports their memory as `N/A`. What varies
between a clean window and a bad one is the megabytes they are holding - 282 MiB in the window
that measured this file's default, against the ~2.1 GB the README records for a contended one.
"""

import os
import subprocess

# A run is refused above this much GPU memory held by others unless --allow-shared-gpu says
# otherwise. 1024 MiB is not a tuning constant: it sits between the 282 MiB of the idle desktop
# contexts measured here and the ~2.1 GB window the README's 4.4x throughput spread came from.
DEFAULT_CEILING_MIB = 1024

# Below this budget a contended window costs minutes, not the 13-18 h a 1M-step Dreamer run takes,
# so the refusal only applies at or above it and the quick benches and tests stay unaffected.
LONG_RUN_STEPS = 200_000

_QUERY = ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,driver_version",
          "--format=csv,noheader,nounits"]
_APPS = ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader"]


def parse_gpu_line(line):
    """`name, used, total, driver` -> dict, or None if the line is not four parseable fields."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 4:
        return None
    name, used, total, driver = parts
    try:
        return {"name": name, "memory_used_mib": int(used), "memory_total_mib": int(total),
                "driver_version": driver}
    except ValueError:
        return None


def parse_context_line(line):
    """`pid, memory` -> (pid, mib or None). Desktop entries report memory as N/A by design."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 2 or not parts[0].isdigit():
        return None
    digits = "".join(ch for ch in parts[1] if ch.isdigit())
    return int(parts[0]), int(digits) if digits else None


def run_nvidia_smi(args):
    """stdout of nvidia-smi, or None. Silent on every failure: this is diagnostics, not a dependency."""
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=15,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
                              if os.name == "nt" else 0)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def read_gpu_window():
    """What the GPU looks like before this process allocates: {gpu, contexts} or None if unknown."""
    out = run_nvidia_smi(_QUERY)
    if not out:
        return None
    gpus = [g for g in (parse_gpu_line(line) for line in out.splitlines()) if g]
    if not gpus:
        return None
    apps_out = run_nvidia_smi(_APPS)
    others = []
    if apps_out:
        for line in apps_out.splitlines():
            parsed = parse_context_line(line)
            if parsed and parsed[0] != os.getpid():
                others.append(parsed)
    return {"gpu": gpus[0], "other_cuda_contexts": len(others),
            "other_context_memory_mib_reported": all(m is not None for _, m in others)}


def check_gpu_window(device_type, allow_shared=False, total_steps=0, log=print,
                     protect_above=LONG_RUN_STEPS):
    """One call a trainer makes: report the window, and refuse to start a LONG run into a bad one.

    Returns the window dict (so the caller can log it as MLflow params once a run exists) or None
    when the check does not apply. The refusal is gated on budget on purpose: what a contended
    window costs is the hours you lose when it dies, and a 5k-step test loses minutes - so the
    suite and the quick benches still run, while a 1M-step run is protected.
    """
    if device_type != "cuda":
        return None
    window = read_gpu_window()
    log(format_window(window))
    long_run = total_steps >= protect_above
    ok, reason = decide_gpu_exclusivity(window, allow_shared)
    if not ok:
        if long_run:
            raise SystemExit(f"[GPU] refusing to start: {reason}")
        reason = (f"{reason} - not enforced here: {total_steps} steps is below the "
                  f"{protect_above}-step long-run threshold, so this run starts anyway")
    log(f"[GPU] {reason}")
    return window


def decide_gpu_exclusivity(window, allow_shared, ceiling_mib=DEFAULT_CEILING_MIB):
    """(ok, reason) for starting a long run in this window. Pure so it can be tested without a GPU."""
    if allow_shared:
        return True, "proceeding on a shared GPU because --allow-shared-gpu was passed"
    if window is None:
        return True, "nvidia-smi unavailable, so the window could not be checked"
    used = window["gpu"]["memory_used_mib"]
    if used > ceiling_mib:
        return False, (f"{used} MiB of GPU memory is already held by other CUDA processes "
                       f"(ceiling {ceiling_mib} MiB, "
                       f"{window['other_cuda_contexts']} contexts). A run started here shares the "
                       "driver queue: this repository measured the same Dreamer build at 15.6 and "
                       "68 env-steps/s in two such windows, and lost a 1M-step run at 84k steps to "
                       "'unspecified launch failure'. Close the desktop GPU users, pass "
                       "--allow-shared-gpu to accept the risk, or run on a host of its own.")
    return True, f"GPU window is clear ({used} MiB held by {window['other_cuda_contexts']} others)"


def format_window(window):
    if not window:
        return "GPU window: unknown (no nvidia-smi)"
    gpu = window["gpu"]
    return (f"GPU window: {gpu['name']} driver {gpu['driver_version']}, "
            f"{gpu['memory_used_mib']}/{gpu['memory_total_mib']} MiB in use by "
            f"{window['other_cuda_contexts']} other CUDA contexts")


def window_params(window):
    """The window as MLflow params, so a published rate can be traced to the GPU state it came from."""
    if not window:
        return {}
    gpu = window["gpu"]
    return {"gpu_name": gpu["name"], "gpu_driver": gpu["driver_version"],
            "gpu_memory_used_mib_before_run": gpu["memory_used_mib"],
            "gpu_memory_total_mib": gpu["memory_total_mib"],
            "gpu_other_cuda_contexts": window["other_cuda_contexts"]}


if __name__ == "__main__":
    # `python -m utils.gpu_window` - print the window a run would start into, and whether it passes.
    w = read_gpu_window()
    print(format_window(w))
    print("ceiling:", DEFAULT_CEILING_MIB, "MiB | long-run threshold:", LONG_RUN_STEPS, "steps")
    if w:
        print(decide_gpu_exclusivity(w, allow_shared=False)[1])
