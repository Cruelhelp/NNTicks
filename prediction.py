import numpy as np
import pickle
import os
import logging
import pandas as pd
from config import PREDICTION_MODE, LEARNING_RATE

class NeuralNetwork:
    def __init__(self, input_size=26, hidden_size=20):  # Adjusted for new features
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weights1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.weights2 = np.random.randn(self.hidden_size, 1) * 0.01
        self.bias1 = np.zeros((1, self.hidden_size))
        self.bias2 = np.zeros((1, 1))
        self.learning_rate = LEARNING_RATE
        self.points = 0
        self.level = 0
        self.points_per_level = 100  # Base points for level 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output):
        self.error = y - output
        self.delta2 = self.error * self.sigmoid_derivative(output)
        self.error_hidden = np.dot(self.delta2, self.weights2.T)
        self.delta1 = self.error_hidden * self.sigmoid_derivative(self.a1)

        self.weights2 += self.learning_rate * np.dot(self.a1.T, self.delta2)
        self.bias2 += self.learning_rate * np.sum(self.delta2, axis=0, keepdims=True)
        self.weights1 += self.learning_rate * np.dot(X.T, self.delta1)
        self.bias1 += self.learning_rate * np.sum(self.delta1, axis=0, keepdims=True)

        weight_deltas = [self.learning_rate * np.dot(self.a1.T, self.delta2), self.learning_rate * np.dot(X.T, self.delta1)]
        return weight_deltas

    def train(self, X, y):
        output = self.forward(X)
        return self.backward(X, y, output)

    def add_points(self, points):
        self.points += points
        required_points = self.points_per_level * (2 ** self.level)
        while self.points >= required_points:
            self.level += 1
            self.learning_rate *= 0.95
            self.points -= required_points
            required_points = self.points_per_level * (2 ** self.level)
            logging.info(f"Level up! New level: {self.level}, Learning Rate: {self.learning_rate:.4f}")
        return required_points - self.points  # Points needed for next level

class PredictionEngine:
    def __init__(self):
        self.nn = NeuralNetwork()
        self.history = []
        self.signal_strength = {"Frequent": 0.5, "Strict": 0.75}
        self.custom_threshold = 0.75
        self.min_trades = 5  # Updated to 5 trades
        self.auto_train = False

    def preprocess(self, prices):
        if len(prices) < 30:
            return None
        prices = np.array(prices[-30:])
        diffs = np.diff(prices)  # 29 differences
        volatility = np.std(prices[-10:])
        if volatility == 0:
            volatility = 1e-6
        scaled_diffs = diffs / volatility
        features = scaled_diffs[-25:]  # Last 25 scaled differences
        features = np.append(features, volatility)  # Add volatility
        mean = np.mean(features)
        std = np.std(features)
        if std == 0:
            std = 1e-6
        normalized = (features - mean) / std
        return normalized.reshape(1, -1)

    def predict(self, prices, mode="Strict"):
        X = self.preprocess(prices)
        if X is None:
            logging.info("Prediction skipped: Not enough data (<30 ticks)")
            return None, None, 0, "NN", "Not enough data"
        try:
            output = self.nn.forward(X)[0][0]
            confidence = output if output > 0.5 else 1 - output
            direction = "Rise" if output > 0.5 else "Fall"
            ticks = 3
            threshold = self.custom_threshold if mode == "Custom" else self.signal_strength[mode]
            thoughts = f"Output: {output:.3f}, Confidence: {confidence:.3f}, Threshold: {threshold:.3f}"
            logging.info(f"NN Prediction Attempt: {thoughts}")
            if confidence < threshold:
                logging.info(f"Prediction rejected: Confidence {confidence:.3f} < {threshold:.3f}")
                return None, None, confidence, "NN", thoughts
            return direction, ticks, confidence, "NN", thoughts
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return None, None, 0, "NN", f"Error: {str(e)}"

    def train(self, prices, direction):
        X = self.preprocess(prices)
        if X is None:
            return []
        y = np.array([[1 if direction == "Rise" else 0]])
        weight_deltas = self.nn.train(X, y)
        self.history.append((prices, direction))
        points_needed = self.nn.add_points(1)
        return weight_deltas

    def train_manual(self, epochs, progress_callback=None):
        if len(self.history) < self.min_trades:
            raise ValueError(f"Need at least {self.min_trades} trades to train (current: {len(self.history)})")
        for epoch in range(epochs):
            np.random.shuffle(self.history)
            total_loss = 0
            for prices, direction in self.history[:100]:
                X = self.preprocess(prices)
                if X is None:
                    continue
                y = np.array([[1 if direction == "Rise" else 0]])
                output = self.nn.forward(X)
                loss = np.mean((y - output) ** 2)
                total_loss += loss
                self.nn.backward(X, y, output)
            avg_loss = total_loss / max(1, min(100, len(self.history)))
            learned_info = f"Level {self.nn.level}, Points {self.nn.points}"
            if progress_callback:
                progress_callback(epoch, avg_loss, learned_info)
            logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    def save_models(self):
        with open(r"C:\Users\ruel.mcneil\Desktop\NNTicks\nn_model.pkl", "wb") as f:
            pickle.dump({"weights1": self.nn.weights1, "weights2": self.nn.weights2,
                         "bias1": self.nn.bias1, "bias2": self.nn.bias2,
                         "points": self.nn.points, "level": self.nn.level,
                         "history": self.history, "min_trades": self.min_trades}, f)
        logging.info("Model saved.")

    def load_models(self):
        try:
            with open(r"C:\Users\ruel.mcneil\Desktop\NNTicks\nn_model.pkl", "rb") as f:
                data = pickle.load(f)
                self.nn.weights1 = data["weights1"]
                self.nn.weights2 = data["weights2"]
                self.nn.bias1 = data["bias1"]
                self.nn.bias2 = data["bias2"]
                self.nn.points = data["points"]
                self.nn.level = data["level"]
                self.history = data["history"]
                self.min_trades = data.get("min_trades", 5)
            logging.info("Model loaded.")
        except FileNotFoundError:
            logging.info("No saved model found, starting fresh.")
        except Exception as e:
            logging.error(f"Model load error: {str(e)}")

    def load_historical_data(self, csv_file):
        df = pd.read_csv(csv_file)
        prices = df['price'].values
        if 'direction' in df.columns:
            directions = df['direction'].values
            for i in range(30, len(prices)):
                price_window = prices[i-30:i]
                X = self.preprocess(price_window)
                if X is None:
                    continue
                y = 1 if directions[i] == "Rise" else 0
                self.nn.train(X, np.array([[y]]))
                self.history.append((price_window, directions[i]))
            logging.info(f"Loaded and trained on {len(self.history)} historical trades.")
        else:
            logging.warning("Historical data lacks 'direction' column; cannot train.")

    def backtest(self, csv_file, tick_count=3, stake=10, mode="Strict"):
        df = pd.read_csv(csv_file)
        prices = df['price'].values
        wins = 0
        losses = 0
        for i in range(30, len(prices) - tick_count):
            price_window = prices[i-30:i]
            X = self.preprocess(price_window)
            if X is None:
                continue
            output = self.nn.forward(X)[0][0]
            confidence = output if output > 0.5 else 1 - output
            threshold = self.custom_threshold if mode == "Custom" else self.signal_strength[mode]
            if confidence < threshold:
                continue
            direction = "Rise" if output > 0.5 else "Fall"
            future_price = prices[i + tick_count]
            actual_direction = "Rise" if future_price > prices[i] else "Fall"
            if direction == actual_direction:
                wins += 1
            else:
                losses += 1
        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        logging.info(f"Backtest results: Wins: {wins}, Losses: {losses}, Win Rate: {win_rate:.2f}%")
        return wins, losses, win_rate