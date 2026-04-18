#!/usr/bin/env python3
"""
Exoplanet Explorer server:
  - Serves static files
  - GET  /api/planets  → NASA TAP proxy (cached)
  - POST /api/chat     → Groq (Llama 3.3 70B) proxy (streaming SSE)
"""
import http.server, urllib.request, urllib.error, json, os, ssl

PORT = int(os.environ.get("PORT", 8743))
GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

NASA_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    "?query=select+pl_name,hostname,ra,dec,sy_dist,pl_rade,pl_masse,"
    "pl_orbper,pl_eqt,discoverymethod,sy_vmag,disc_year,st_teff,st_lum"
    "+from+ps+where+default_flag=1+and+ra+is+not+null"
    "+and+dec+is+not+null+and+sy_dist+is+not+null"
    "&format=json"
)

SYSTEM_PROMPT = """\
You are an expert exoplanet science assistant embedded in an interactive 3D galaxy \
explorer showing all 6,000+ confirmed exoplanets from NASA's archive.

You help users understand:
- What makes planets Earth-like (ESI score based on radius and equilibrium temperature)
- The science behind discovery methods: Transit, Radial Velocity, Direct Imaging, \
Microlensing, Transit Timing Variations, Pulsar Timing, Astrometry
- Characteristics of specific planetary systems and how they compare to our Solar System
- Famous discoveries and their historical significance
- The staggering scale of space (distances, travel times, star types)

Be enthusiastic, precise, and make astrophysics genuinely exciting and accessible. \
Use concrete numbers and vivid comparisons. Keep answers focused and conversational — \
2-4 paragraphs max unless asked to go deeper. No bullet-point walls; write like a \
brilliant friend who happens to know everything about exoplanets."""

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

PLANET_CACHE = None


class Handler(http.server.SimpleHTTPRequestHandler):

    # ── CORS preflight ──────────────────────────────
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    # ── GET ─────────────────────────────────────────
    def do_GET(self):
        if self.path.startswith("/api/planets"):
            self.serve_planets()
        else:
            super().do_GET()

    # ── POST ────────────────────────────────────────
    def do_POST(self):
        if self.path.startswith("/api/chat"):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            self.serve_chat(body)
        else:
            self.send_response(404)
            self.end_headers()

    # ── Planet proxy ────────────────────────────────
    def serve_planets(self):
        global PLANET_CACHE
        if PLANET_CACHE is None:
            try:
                req = urllib.request.Request(
                    NASA_URL, headers={"User-Agent": "ExoplanetExplorer/1.0"})
                with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as r:
                    PLANET_CACHE = r.read()
            except Exception as e:
                self.send_response(502)
                self._cors()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
                return

        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(PLANET_CACHE)

    # ── Groq chat proxy (streaming SSE) ─────────────
    def serve_chat(self, body):
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def emit(data: dict):
            self.wfile.write(("data: " + json.dumps(data) + "\n\n").encode())
            self.wfile.flush()

        if not GROQ_KEY:
            emit({"error": (
                "No API key found. In your terminal run:\n"
                "  export GROQ_API_KEY=gsk-...\n"
                "then restart server.py."
            )})
            return

        try:
            payload_in = json.loads(body)
            messages   = payload_in.get("messages", [])
            context    = payload_in.get("context", "")

            system = SYSTEM_PROMPT
            if context:
                system += f"\n\n--- CURRENT EXPLORER CONTEXT ---\n{context}"

            # Groq uses OpenAI-compatible format: system goes in messages array
            groq_messages = [{"role": "system", "content": system}] + messages

            payload_out = json.dumps({
                "model": "llama-3.3-70b-versatile",
                "max_tokens": 1024,
                "messages": groq_messages,
                "stream": True,
            }).encode()

            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=payload_out,
                headers={
                    "Authorization": f"Bearer {GROQ_KEY}",
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                    "User-Agent": "ExoplanetExplorer/1.0",
                },
            )
            with urllib.request.urlopen(req, timeout=60, context=SSL_CTX) as r:
                for raw in r:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        ev = json.loads(chunk)
                        text = ev["choices"][0]["delta"].get("content", "")
                        if text:
                            emit({"text": text})
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass
            emit({"done": True})

        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", errors="replace")
                emit({"error": f"HTTP {e.code}: {body}"})
            except Exception:
                emit({"error": str(e)})
        except Exception as e:
            try:
                emit({"error": str(e)})
            except Exception:
                pass

    # ── Helpers ─────────────────────────────────────
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, fmt, *args):
        path = args[0] if args else ""
        if "/api/" in path:
            print(f"  [{args[1]}] {path}")
        else:
            super().log_message(fmt, *args)


os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"Exoplanet Explorer → http://localhost:{PORT}/exoplanet_explorer.html")
if not GROQ_KEY:
    print("  ⚠  No GROQ_API_KEY — chat will prompt you to set it.")
    print("     Run: export GROQ_API_KEY=gsk-...")
with http.server.ThreadingHTTPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()
