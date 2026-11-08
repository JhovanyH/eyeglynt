"""Runs the dashboard for testing on a laptop.

On the Pi, setup_dashboard.sh runs it with gunicorn on https://<pi-ip>:5000
using the certificate in certs/. Here, the same certificate is used if
certs/ exists; otherwise a temporary one is made (needs the "cryptography"
package from requirements.txt). The browser warns once that the
certificate is self-made: choose Advanced, then Proceed.
Open https://localhost:5000 (with the s).

To test without HTTPS on a laptop, add the line  HTTPS=0  to .env and
open http://localhost:5000 instead.
"""
import os

from dotenv import load_dotenv

from app import create_app

load_dotenv()
app = create_app()

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    cert = os.path.join(here, "certs", "cert.pem")
    key = os.path.join(here, "certs", "key.pem")
    if os.getenv("HTTPS", "1") == "0":
        ssl_context = None
        print("HTTPS is off (HTTPS=0 in .env): open http://localhost:5000")
    else:
        ssl_context = (cert, key) if os.path.exists(cert) and os.path.exists(key) else "adhoc"
        print("Open https://localhost:5000 (with the s). If the browser warns, choose Advanced, then Proceed.")
    # 0.0.0.0 so phones/laptops on the same Wi-Fi can reach this computer too
    app.run(host="0.0.0.0", port=5000, debug=False, ssl_context=ssl_context)
