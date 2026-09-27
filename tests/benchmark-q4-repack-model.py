#!/usr/bin/env python3
"""Fixed-token Flash-Next HTTP workload; restart with Q4 opt-in before each run.

Run labels in A/B/B/A order against the same binary and server configuration.
Outputs include exact prompt/output tokens, timings and speculative statistics.
No service lifecycle, model mutation or credentials are handled here.
"""
import argparse
import json
import math
import pathlib
import time
import urllib.request


def request(base, path, data=None):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(base + path, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8089")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--generation-only", action="store_true", help="MTP code/prose TG workloads")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    texts = {
        "code": "Write a robust Python LRU cache with unit tests. Explain thread safety and eviction.\n",
        "prose": "Explain how a city can prepare for summer heat while protecting vulnerable residents.\n",
    }
    fixtures_path = args.output / "prompts.json"
    if fixtures_path.exists():
        fixtures = json.loads(fixtures_path.read_text())
    else:
        fixtures = {}
        for name, text in texts.items():
            fixtures[name] = request(args.url, "/tokenize", {"content": text * 800,
                                                           "add_special": True})["tokens"]
            assert len(fixtures[name]) >= 4096
        fixtures_path.write_text(json.dumps(fixtures))
    cases = [("code", 64, 32), ("prose", 64, 32)] if args.smoke else [
        ("code", 512, 1), ("code", 4096, 1),
        ("code", 512, 128), ("prose", 512, 512),
    ]
    if args.generation_only:
        cases = [("code", 512, 128), ("prose", 512, 512)]
    for index, (kind, pp, tg) in enumerate(cases):
        payload = {"prompt": fixtures[kind][:pp], "n_predict": tg,
                   "temperature": 0, "seed": 42, "cache_prompt": False,
                   "ignore_eos": True, "return_tokens": True}
        start = time.monotonic()
        response = request(args.url, "/completion", payload)
        for value in response["timings"].values():
            if isinstance(value, (int, float)):
                assert math.isfinite(value), response["timings"]
        assert response["tokens_predicted"] == tg, response
        result = {"label": args.label, "kind": kind, "pp": pp, "tg": tg,
                  "wall_seconds": time.monotonic() - start, "request": payload,
                  "response": response}
        path = args.output / f"{args.label}-{index}-{kind}-pp{pp}-tg{tg}.json"
        path.write_text(json.dumps(result, indent=2))
        print(json.dumps({"file": str(path), "timings": response["timings"]}), flush=True)


if __name__ == "__main__":
    main()
