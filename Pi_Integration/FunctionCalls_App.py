# Function Calls for Mobile App (Flask video stream + email alerts)

import os
import sys
import logging
import threading

_PI_INTEGRATION_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_PI_INTEGRATION_DIR)
_APP_DIR = os.path.join(_PROJECT_ROOT, 'AppMobile')

if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import AppMobileFire as AppMobile

logger = logging.getLogger(__name__)


class MobileApp:
    """
    Wrapper around AppMobile.py providing Flask video streaming,
    mDNS advertisement, and email alert capabilities.
    """

    def __init__(self):
        self._server_running = False
        self._zeroconf = None
        self._mdns_info = None
        self._server_ip = None

    def start_server(self) -> None:
        """Start the Flask MJPEG server and mDNS advertisement in the background."""
        if self._server_running:
            logger.warning("Mobile app server already running")
            return

        AppMobile.start_flask_in_thread(host='0.0.0.0', port=AppMobile.FLASK_PORT)

        self._zeroconf, self._mdns_info = AppMobile.register_mdns_service(
            name=AppMobile.MDNS_NAME, port=AppMobile.FLASK_PORT
        )

        self._server_ip = AppMobile.get_host_ip()
        self._server_running = True
        logger.info(f"Mobile app server started at http://{self._server_ip}:{AppMobile.FLASK_PORT}/")

    def send_alert(self, confidence: float) -> None:
        """
        Send an email notification with the video stream URL.

        Args:
            confidence: Cry detection confidence to include in the alert.
        """
        if not self._server_running:
            logger.warning("Cannot send alert — server not running")
            return

        if not AppMobile.SMTP_PASSWORD:
            logger.warning("SMTP_PASSWORD not set; skipping email alert")
            return

        logger.info(f"Sending alert email (confidence={confidence:.1%})...")
        AppMobile.send_stream_email(
            sender_email=AppMobile.SENDER_EMAIL,
            receiver_email=AppMobile.RECEIVER_EMAIL,
            password=AppMobile.SMTP_PASSWORD,
            server_ip=self._server_ip,
            port=AppMobile.FLASK_PORT,
        )

        # Update polling status so the Android app can also detect the alert
        AppMobile.notification_status["should_notify"] = True
        AppMobile.notification_status["confidence_score"] = confidence
        AppMobile.notification_status["message"] = f"BABY MONITOR ALERT! Confidence: {confidence:.1%}"

    @property
    def stream_url(self) -> str:
        """Return the video stream URL (or empty string if not started)."""
        if self._server_ip:
            return f"http://{self._server_ip}:{AppMobile.FLASK_PORT}/"
        return ""

    def stop(self) -> None:
        """Clean up mDNS registration and release the camera."""
        if self._zeroconf and self._mdns_info:
            try:
                self._zeroconf.unregister_service(self._mdns_info)
                self._zeroconf.close()
                logger.info("mDNS service unregistered")
            except Exception as e:
                logger.error(f"Error during mDNS cleanup: {e}")

        try:
            if AppMobile.camera is not None:
                AppMobile.camera.release()
                logger.info("Camera released")
        except Exception as e:
            logger.error(f"Error releasing camera: {e}")

        self._server_running = False
        logger.info("MobileApp stopped")
