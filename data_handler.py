import websocket
import json
import logging
import time
import threading
from config import WS_URL, SYMBOL, API_TOKEN

class DataHandler:
    def __init__(self, api_url=WS_URL, api_key=API_TOKEN):
        self.api_url = api_url
        self.api_key = api_key
        self.ws = None
        self.prices = []
        self.timestamped_prices = []
        self.running = False
        self.ws_thread = None
        self.last_price = None

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            if "error" in data:
                logging.error(f"WebSocket error: {data['error']}")
            elif "authorize" in data:
                if data.get("authorize", {}).get("status") == "success":
                    logging.info("Authorization successful")
                else:
                    logging.error("Authorization failed")
            elif "tick" in data:
                price = float(data["tick"]["quote"])
                timestamp = float(data["tick"]["epoch"])
                if self.last_price is not None and abs(price - self.last_price) / self.last_price > 0.1:
                    logging.warning(f"Large price jump detected: {self.last_price} to {price}")
                self.last_price = price
                if len(self.timestamped_prices) > 0 and timestamp <= self.timestamped_prices[-1][0]:
                    logging.warning(f"Non-increasing timestamp: {timestamp} <= {self.timestamped_prices[-1][0]}")
                self.prices.append(price)
                self.timestamped_prices.append((timestamp, price))
                if len(self.prices) > 500:
                    self.prices.pop(0)
                if len(self.timestamped_prices) > 500:
                    self.timestamped_prices.pop(0)
                logging.info(f"Received tick: {price} at {timestamp}")
            else:
                logging.info(f"Received unknown message: {data}")
        except Exception as e:
            logging.error(f"Message parsing error: {e}")

    def on_error(self, ws, error):
        logging.error(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        logging.info(f"WebSocket closed: {close_msg} (code: {close_status_code})")

    def on_open(self, ws):
        logging.info("WebSocket connection opened.")
        subscription = {"ticks": SYMBOL, "subscribe": 1}
        ws.send(json.dumps(subscription))
        if self.api_key:
            ws.send(json.dumps({"authorize": self.api_key}))

    def ws_run(self):
        while self.running:
            try:
                self.ws = websocket.WebSocketApp(
                    self.api_url,
                    on_open=self.on_open,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close
                )
                self.ws.run_forever()
                if not self.running:
                    break
                logging.info("WebSocket connection closed, reconnecting in 5 seconds...")
                time.sleep(5)
            except Exception as e:
                logging.error(f"WebSocket connection error: {e}")
                time.sleep(5)

    def connect(self):
        if self.running:
            return
        self.running = True
        self.ws_thread = threading.Thread(target=self.ws_run)
        self.ws_thread.start()

    def stop(self):
        if self.running:
            self.running = False
            if self.ws:
                self.ws.close()
            if self.ws_thread:
                self.ws_thread.join()
            logging.info("WebSocket stopped.")

    def get_prices(self):
        return self.prices

    def get_timestamped_prices(self):
        return self.timestamped_prices