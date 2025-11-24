#!/usr/bin/env python3
"""Generate soxr reference data JSON file."""

import subprocess
import json
import re
import sys

def run_tool(args):
    """Run a test tool and return stdout."""
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return result.stdout
    except Exception as e:
        print(f"Error running {args}: {e}", file=sys.stderr)
        return ""

def parse_value(output, pattern):
    """Extract a numeric value from output using regex."""
    match = re.search(pattern, output)
    if match:
        try:
            return float(match.group(1))
        except:
            return 0.0
    return 0.0

data = {
    "antialiasing": {},
    "quality": {}
}

# Anti-aliasing upsampling
for in_rate, out_rate, sig in [
    (48000, 96000, "noise"),
    (48000, 96000, "multitone"),
    (44100, 88200, "noise"),
    (44100, 96000, "noise"),
]:
    out = run_tool(["./test_antialiasing", str(in_rate), str(out_rate), sig])
    att = parse_value(out, r"STOPBAND ATTENUATION:\s*([-\d.]+)")
    key = f"{in_rate}_{out_rate}_{sig}"
    data["antialiasing"][key] = att
    print(f"antialiasing {key}: {att}", file=sys.stderr)

# Anti-aliasing downsampling
for in_rate, out_rate in [
    (48000, 32000),
    (48000, 44100),
    (96000, 48000),
]:
    out = run_tool(["./test_antialiasing", str(in_rate), str(out_rate), "alias_tones"])
    att = parse_value(out, r"ANTI-ALIASING ATTENUATION:\s*([-\d.]+)")
    key = f"{in_rate}_{out_rate}_alias_tones"
    data["antialiasing"][key] = att
    print(f"antialiasing {key}: {att}", file=sys.stderr)

# Passband ripple
for in_rate, out_rate in [
    (44100, 48000),
    (48000, 44100),
    (48000, 96000),
    (96000, 48000),
    (48000, 32000),
]:
    out = run_tool(["./test_quality", str(in_rate), str(out_rate), "ripple"])
    ripple = parse_value(out, r"ripple\s*=\s*([-\d.]+)")
    max_dev = parse_value(out, r"max_deviation\s*=\s*([-\d.]+)")
    min_dev = parse_value(out, r"min_deviation\s*=\s*([-\d.]+)")
    key = f"ripple_{in_rate}_{out_rate}"
    data["quality"][key] = {"ripple": ripple, "max_dev": max_dev, "min_dev": min_dev}
    print(f"quality {key}: ripple={ripple}", file=sys.stderr)

# THD
for in_rate, out_rate, freq in [
    (44100, 48000, 1000),
    (48000, 44100, 1000),
    (48000, 96000, 1000),
    (96000, 48000, 1000),
    (48000, 32000, 1000),
    (44100, 48000, 10000),
    (48000, 44100, 10000),
]:
    out = run_tool(["./test_quality", str(in_rate), str(out_rate), f"thd:{freq}"])
    thd_db = parse_value(out, r"thd_db\s*=\s*([-\d.]+)")
    thd_pct = parse_value(out, r"thd_percent\s*=\s*([-\d.]+)")
    key = f"thd_{in_rate}_{out_rate}_{freq}"
    data["quality"][key] = {"thd_db": thd_db, "thd_percent": thd_pct}
    print(f"quality {key}: thd_db={thd_db}", file=sys.stderr)

# SNR
for in_rate, out_rate in [
    (44100, 48000),
    (48000, 44100),
    (48000, 96000),
    (96000, 48000),
    (48000, 32000),
]:
    out = run_tool(["./test_quality", str(in_rate), str(out_rate), "snr:1000"])
    snr = parse_value(out, r"snr_db\s*=\s*([-\d.]+)")
    key = f"snr_{in_rate}_{out_rate}"
    data["quality"][key] = snr
    print(f"quality {key}: snr={snr}", file=sys.stderr)

# Impulse response
for in_rate, out_rate in [
    (44100, 48000),
    (48000, 44100),
    (48000, 96000),
    (96000, 48000),
    (48000, 32000),
]:
    out = run_tool(["./test_quality", str(in_rate), str(out_rate), "impulse"])
    pre = parse_value(out, r"pre_ringing_db\s*=\s*([-\d.]+)")
    post = parse_value(out, r"post_ringing_db\s*=\s*([-\d.]+)")
    ringout = int(parse_value(out, r"ringout_samples\s*=\s*(\d+)"))
    key = f"impulse_{in_rate}_{out_rate}"
    data["quality"][key] = {"pre_ringing_db": pre, "post_ringing_db": post, "ringout_samples": ringout}
    print(f"quality {key}: pre={pre}, post={post}", file=sys.stderr)

print(json.dumps(data, indent=2))
