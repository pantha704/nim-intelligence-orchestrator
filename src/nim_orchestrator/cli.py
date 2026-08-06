import argparse
import asyncio
import json
import sys

from .api import handle_intelligence_request, run_benchmark
from .config import load_settings
from .router_client import RouterClient


def main():
    parser = argparse.ArgumentParser(
        prog="nim-orch",
        description="NIM Intelligence Orchestrator — multi-model intelligence layer",
    )
    subparsers = parser.add_subparsers(dest="command")

    ask_parser = subparsers.add_parser("ask", help="Ask a question through the intelligence pipeline")
    ask_parser.add_argument("prompt", help="Question/prompt to process")
    ask_parser.add_argument("--mode", choices=["auto", "single", "full", "dag"], default="auto")

    bench_parser = subparsers.add_parser("bench", help="Run benchmark comparing modes")
    bench_parser.add_argument("--output", "-o", default=None, help="Save results to file")

    serve_parser = subparsers.add_parser("serve", help="Start the orchestrator API server")
    serve_parser.add_argument("--host", default=None)
    serve_parser.add_argument("--port", type=int, default=None)

    subparsers.add_parser("health", help="Check router connectivity")

    args = parser.parse_args()

    if args.command == "ask":
        force_mode = None
        if args.mode == "single":
            force_mode = "single"
        elif args.mode == "full":
            force_mode = "full"
        elif args.mode == "dag":
            force_mode = "dag"

        async def _ask():
            settings = load_settings()
            if not settings.router_api_key:
                print("ERROR: No API key found. Check config/orchestrator.env", file=sys.stderr)
                sys.exit(1)
            if not settings.candidates:
                print("ERROR: No candidates configured. Check config/orchestrator.yaml", file=sys.stderr)
                sys.exit(1)

            client = RouterClient(settings.router_base_url, settings.router_api_key)
            result = await handle_intelligence_request(
                client, settings, args.prompt, force_mode=force_mode,
            )
            await client.close()
            return result

        result = asyncio.run(_ask())

        if result.get("error") and not result.get("answer"):
            print(f"Rejected: {result['error']}", file=sys.stderr)
            sys.exit(1)

        print(result["answer"])
        if result.get("pipeline_trace"):
            print("\n--- Pipeline Trace ---", file=sys.stderr)
            for line in result["pipeline_trace"]:
                print(f"  {line}", file=sys.stderr)
            print(f"  Mode: {result['mode']} | Latency: {result.get('latency_ms', 0):.0f}ms", file=sys.stderr)

    elif args.command == "bench":
        settings = load_settings()
        print("Running benchmark (this may take several minutes)...", file=sys.stderr)
        results = asyncio.run(run_benchmark(settings))

        print("\n" + "=" * 60)
        print("  Benchmark Summary")
        print("=" * 60)
        for mode, stats in results["summary"].items():
            score = stats["mean_score"]
            latency = stats["mean_latency_ms"]
            print(f"  {mode:20s}  score={score:.1%}  latency={latency:.0f}ms  errors={stats['error_count']}/{stats['total']}")
        print("=" * 60)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nFull results saved to {args.output}", file=sys.stderr)

    elif args.command == "serve":
        settings = load_settings()
        host = args.host or settings.orchestrator_host
        port = args.port or settings.orchestrator_port
        _run_server(settings, host, port)

    elif args.command == "health":
        settings = load_settings()

        async def _health():
            client = RouterClient(settings.router_base_url, settings.router_api_key)
            healthy = await client.health()
            models = await client.models() if healthy else []
            await client.close()
            return healthy, models

        healthy, models = asyncio.run(_health())
        if healthy:
            print(f"OK: Router at {settings.router_base_url} is healthy")
            print(f"Models: {', '.join(models)}")
        else:
            print(f"FAIL: Router at {settings.router_base_url} is not responding")
            sys.exit(1)

    else:
        parser.print_help()


def _run_server(settings, host: str, port: int):
    import http.server
    import json

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self._json_response({"status": "ok"})
            else:
                self._json_response({"error": "not found"}, 404)

        def do_POST(self):
            if self.path == "/v1/intelligence":
                self._handle_intelligence()
            else:
                self._json_response({"error": "not found"}, 404)

        def _handle_intelligence(self):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                req = json.loads(body)
            except json.JSONDecodeError:
                self._json_response({"error": "invalid JSON"}, 400)
                return

            prompt = req.get("prompt", "")
            force_mode = req.get("mode")

            if not prompt:
                self._json_response({"error": "missing 'prompt'"}, 400)
                return

            async def _run():
                client = RouterClient(settings.router_base_url, settings.router_api_key)
                result = await handle_intelligence_request(client, settings, prompt, force_mode)
                await client.close()
                return result

            try:
                result = asyncio.run(_run())
                self._json_response(result)
            except Exception as e:
                self._json_response({"error": str(e)[:500]}, 500)

        def _json_response(self, data, status=200):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            body = json.dumps(data, indent=2).encode()
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            import sys
            print(f"[orchestrator] {args[0]} {args[1]} {args[2]}", file=sys.stderr)

    server = http.server.HTTPServer((host, port), Handler)
    print(f"NIM Intelligence Orchestrator listening on http://{host}:{port}", file=sys.stderr)
    print("  POST /v1/intelligence  — run intelligence pipeline", file=sys.stderr)
    print("  GET  /health            — health check", file=sys.stderr)
    server.serve_forever()


if __name__ == "__main__":
    main()
