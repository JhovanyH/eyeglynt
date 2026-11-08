"""Runs the dashboard on HTTPS for testing on a laptop.

On the Pi, setup_dashboard.sh runs it with gunicorn on https://<pi-ip>:5000
using the certificate in certs/. Here, the same certificate is used if
certs/ exists; otherwise a temporary one is made (needs the "cryptography"
package from requirements.txt). The browser will warn once that the
certificate is self-made: choose Advanced, then Proceed.
"""
import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    cert = os.path.join(here, "certs", "cert.pem")
    key = os.path.join(here, "certs", "key.pem")
    ssl_context = (cert, key) if os.path.exists(cert) and os.path.exists(key) else "adhoc"
    # 0.0.0.0 so phones/laptops on the same Wi-Fi can open https://<this-computer-ip>:5000
    app.run(host="0.0.0.0", port=5000, debug=False, ssl_context=ssl_context)
