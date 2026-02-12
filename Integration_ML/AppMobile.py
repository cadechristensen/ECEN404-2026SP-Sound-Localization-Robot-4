#!/usr/bin/env python3

#python C:\Users\Cade\Downloads\AppMobile.py
#IP: 165.91.10.243
import os
import socket
import threading
import time
import signal
import sys
from email.mime.text import MIMEText
import smtplib

from flask import Flask, Response
import cv2

from zeroconf import Zeroconf, ServiceInfo

# Configuration (env-driven)

SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "soundlocal4@gmail.com")
RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL", "cadechristensen@tamu.edu")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "jqoo xinq appe ukge")
MDNS_NAME = os.environ.get("MDNS_NAME", "BabyMonitor")
FLASK_PORT = int(os.environ.get("FLASK_PORT", "5000"))
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))

if SMTP_PASSWORD is None:
    print("WARNING: SMTP_PASSWORD not set. Email sending will fail until you set SMTP_PASSWORD env var.")

# Flask app & camera
app = Flask(__name__)
camera = cv2.VideoCapture(CAMERA_INDEX)

def generate_frames():

    while True:
        success, frame = camera.read()
        if not success:
            time.sleep(0.1)
            continue
        # encode as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <html>
      <head>
        <title>Live Baby Monitor</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
          body { background: #87CEEB; color: #fff; font-family: Arial, sans-serif; text-align: center; }
          .container { margin-top: 2em; padding: 2em; background: #242b39; border-radius: 12px; display: inline-block; }
          #video-frame { border: 4px solid #6683d2; border-radius: 8px; margin-top: 1em; box-shadow: 0 4px 18px rgba(0,0,0,0.25); width: 80vw; max-width: 720px; height: auto; max-height: 150vh; object-fit: contain; }
          h1 { margin-bottom: 0; font-size: 2.2em; }
          p { margin-top: 1em; color: #b9c6ee; }
        </style>
      </head>
      <body>
        <div class="container">
          <h1>Baby Monitor</h1>
          <p>👶</p>
          <img id="video-frame" src="/video" />
        </div>
      </body>
    </html>
    '''

# Email sending
def send_stream_email(sender_email, receiver_email, password, server_ip, port=FLASK_PORT):
    stream_url = f"http://{server_ip}:{port}/"
    message = MIMEText(f"The robot has located the baby: {stream_url}")
    message["Subject"] = "BABY MONITOR ALERT"
    message["From"] = sender_email
    message["To"] = receiver_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        print(f"Alert email sent to {receiver_email}")
    except Exception as e:
        print("Failed to send email:", e)

# mDNS / zeroconf advertisement
def get_host_ip():

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # connect to an external DNS server (Google) — doesn't send data but lets us read local IP
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        # fallback
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"

def register_mdns_service(name=MDNS_NAME, port=FLASK_PORT, properties=None):
    zeroconf = Zeroconf()
    hostname = socket.gethostname()
    host_ip = get_host_ip()
    service_name = f"{name}._http._tcp.local."
    info = ServiceInfo(
        type_="_http._tcp.local.",
        name=service_name,
        addresses=[socket.inet_aton(host_ip)],
        port=port,
        properties=properties or {},
        server=f"{hostname}.local."
    )
    try:
        zeroconf.register_service(info)
        print(f"Registered mDNS service {service_name} at {host_ip}:{port}")
    except Exception as e:
        print("Failed to register mDNS service:", e)
        zeroconf.close()
        return None, None
    return zeroconf, info

# Startup
flask_thread = None
zeroconf_obj = None
mdns_info = None
shutdown_requested = False

def start_flask_in_thread(host='0.0.0.0', port=FLASK_PORT):
    global flask_thread
    def run():
        app.run(host=host, port=port, threaded=True, use_reloader=False)
    flask_thread = threading.Thread(target=run, daemon=True)
    flask_thread.start()
    print(f"Flask started in background thread on {host}:{port}")

def start_services_and_notify():
    """
    Start Flask, advertise via mDNS, then send an email notifying the receiver.
    """
    global zeroconf_obj, mdns_info

    # RECORD START TIME
    start_time = time.time()

    # Start Flask
    start_flask_in_thread(host='0.0.0.0', port=FLASK_PORT)
    time.sleep(1.0)

    zeroconf_obj, mdns_info = register_mdns_service(name=MDNS_NAME, port=FLASK_PORT)
    server_ip = get_host_ip()
    print(f"Server IP (detected): {server_ip}")

    if SMTP_PASSWORD:
        try:
            send_stream_email(SENDER_EMAIL, RECEIVER_EMAIL, SMTP_PASSWORD, server_ip, port=FLASK_PORT)
        except Exception as e:
            print("Error sending notification email:", e)
    else:
        print("SMTP_PASSWORD not set; skipping email notification.")

    # CALCULATE AND PRINT ELAPSED TIME
    elapsed = time.time() - start_time
    print(f"Device/server connection time: {elapsed:.2f} seconds")

# -----------------------
# Graceful shutdown handling
# -----------------------
def shutdown(signum=None, frame=None):
    global shutdown_requested, zeroconf_obj, mdns_info, camera
    if shutdown_requested:
        return
    print("Shutdown requested. Cleaning up...")
    shutdown_requested = True
    try:
        # unregister mdns
        if zeroconf_obj and mdns_info:
            try:
                zeroconf_obj.unregister_service(mdns_info)
                print("Unregistered mDNS service")
            except Exception as e:
                print("Error unregistering mDNS service:", e)
            try:
                zeroconf_obj.close()
            except Exception:
                pass
    except Exception as e:
        print("Error during mDNS cleanup:", e)

    try:
        # release camera
        if camera is not None:
            camera.release()
            print("Camera released")
    except Exception as e:
        print("Error releasing camera:", e)

    # exit process
    print("Exiting.")
    sys.exit(0)

# Register signal handlers for Ctrl+C and graceful termination
signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

# -----------------------
# Main entrypoint
# -----------------------
if __name__ == '__main__':
    print("Starting Flask MJPEG server with mDNS advertisement...")
    start_services_and_notify()
    # Keep the main thread alive while background threads run
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown()
