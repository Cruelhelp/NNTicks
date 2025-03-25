import time
from data_handler import DataHandler
from prediction import PredictionEngine
from config import PREDICTION_MODE
import logging

def main():
    logging.info("Starting NNTicks Phase 2 Test - Live Streaming with Strong Signals...")
    data_handler = DataHandler()
    predictor = PredictionEngine()

    # Start live tick streaming
    data_handler.connect()

    # Wait for initial data, predict only on strong signals for 60 seconds
    start_time = time.time()
    while time.time() - start_time < 60:  # Run for 60 seconds
        prices = data_handler.get_prices()
        if len(prices) >= 26:  # Need 26 ticks for meaningful features
            direction, ticks, confidence, model = predictor.predict(prices, PREDICTION_MODE)
            if direction:  # Only log if prediction meets threshold
                logging.info(f"Collected {len(prices)} prices, latest: {prices[-1]}")
                logging.info(f"Prediction: {direction} in {ticks} ticks [Confidence: {confidence:.3f}] [{model}]")
        time.sleep(1)  # Check every second

    # Cleanup
    data_handler.stop()
    logging.info("Phase 2 Test Complete.")

if __name__ == "__main__":
    main()