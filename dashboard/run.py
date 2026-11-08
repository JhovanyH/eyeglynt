"""Development server.  On the Pi use:  gunicorn -w 2 -b 0.0.0.0:5000 run:app"""
from app import create_app

app = create_app()

if __name__ == "__main__":
    # 0.0.0.0 so phones/laptops on the same Wi-Fi can open http://<pi-ip>:5000
    app.run(host="0.0.0.0", port=5000, debug=False)
