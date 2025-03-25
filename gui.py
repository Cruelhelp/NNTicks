import sys
import time
import pickle
import json
import os
import winsound
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QProgressBar, QTabWidget, QComboBox, QPushButton, QTextEdit, QMessageBox,
                             QSplashScreen, QLineEdit, QSpinBox, QCheckBox, QDoubleSpinBox, QFileDialog)
from PyQt6.QtCore import QTimer, QThread, pyqtSignal, Qt, QUrl, QPropertyAnimation, QEasingCurve
from PyQt6.QtMultimedia import QSoundEffect
from PyQt6.QtGui import QPixmap, QCursor, QScreen
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import mplfinance as mpf
import pandas as pd
from data_handler import DataHandler
from prediction import PredictionEngine
from config import PREDICTION_MODE, WS_URL, API_TOKEN
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import logging
import traceback
import mplcursors

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", filename=r"C:\Users\ruel.mcneil\Desktop\NNTicks\nnticks.log")

class TickWorker(QThread):
    tick_received = pyqtSignal(float, float)
    prediction_made = pyqtSignal(str, int, float, str, str, float)
    entry_signal = pyqtSignal(float)
    countdown_signal = pyqtSignal(int)
    outcome_received = pyqtSignal(str, float, float, str)
    error_signal = pyqtSignal(str)
    training_progress = pyqtSignal(int, float, str)
    prediction_log = pyqtSignal(str)

    def __init__(self, tick_count=3, mode="Strict", api_url=WS_URL, api_key=API_TOKEN):
        super().__init__()
        self.data_handler = DataHandler(api_url, api_key)
        self.predictor = PredictionEngine()
        self.tick_count = tick_count
        self.mode = mode
        self.running = False
        self.last_tick_time = 0
        self.last_pred_time = 0
        self.active_trade = False
        self.entry_time = None
        self.entry_price = None
        self.direction = None
        self.tick_count_passed = 0
        self.cooldown = {"Frequent": 10, "Strict": 30, "Custom": 30}
        logging.info(f"TickWorker initialized with API URL: {api_url}")

    def set_tick_count(self, count):
        self.tick_count = count
        logging.info(f"Tick count updated to: {count}")

    def set_mode(self, mode):
        self.mode = mode
        logging.info(f"Prediction mode updated to: {mode}")

    def run(self):
        try:
            self.data_handler.connect()
            while self.running:
                try:
                    prices = self.data_handler.get_prices()
                    timestamped_prices = self.data_handler.get_timestamped_prices()
                    if not timestamped_prices:
                        time.sleep(0.05)
                        continue

                    latest_time, latest_price = timestamped_prices[-1]
                    if time.time() - self.last_tick_time >= 1.0:
                        self.tick_received.emit(latest_price, latest_time)
                        self.last_tick_time = time.time()

                    if (len(prices) >= 30 and
                        time.time() - self.last_pred_time > self.cooldown.get(self.mode, 30) and
                        not self.active_trade):
                        logging.info(f"Attempting prediction: {len(prices)} prices, Mode: {self.mode}, Cooldown: {time.time() - self.last_pred_time:.1f}/{self.cooldown.get(self.mode, 30)}s")
                        direction, ticks, confidence, model, thoughts = self.predictor.predict(prices, self.mode)
                        self.prediction_log.emit(f"[{time.strftime('%H:%M:%S')}] {thoughts}")
                        logging.info(f"Prediction result: Direction={direction}, Confidence={confidence:.3f}, Thoughts={thoughts}")
                        if direction:
                            pred_time = latest_time
                            pred_price = latest_price
                            self.prediction_made.emit(direction, self.tick_count, confidence, model, thoughts, pred_price)
                            self.last_pred_time = time.time()
                            self.active_trade = True
                            logging.info(f"NN Prediction: {direction} in {ticks} ticks [Confidence: {confidence:.3f}]")

                    if self.active_trade and self.entry_time is not None:
                        ticks_since_entry = sum(1 for t, _ in timestamped_prices if t > self.entry_time)
                        if ticks_since_entry > self.tick_count_passed and self.tick_count_passed < self.tick_count:
                            self.tick_count_passed += 1
                            self.countdown_signal.emit(self.tick_count_passed)
                            if self.tick_count_passed == self.tick_count:
                                end_price = timestamped_prices[-1][1]
                                outcome = "Win" if (self.direction == "Rise" and end_price > self.entry_price) else "Loss"
                                self.outcome_received.emit(self.direction, self.entry_price, end_price, outcome)
                                entry_index = next(i for i, p in enumerate(timestamped_prices) if p[0] == self.entry_time)
                                train_dir = self.direction if outcome == "Win" else ("Fall" if self.direction == "Rise" else "Rise")
                                weight_deltas = self.predictor.train(prices[:entry_index + 1], train_dir)
                                self.active_trade = False
                                self.entry_time = None
                                self.entry_price = None
                                self.direction = None
                                self.tick_count_passed = 0
                                if self.predictor.auto_train:
                                    self.train_manual(500)

                    if self.active_trade and time.time() - self.last_pred_time >= 10 and self.entry_time is None:
                        self.entry_time = timestamped_prices[-1][0]
                        self.entry_price = timestamped_prices[-1][1]
                        self.direction = self.predictor.predict(prices, self.mode)[0]
                        self.tick_count_passed = 0
                        self.entry_signal.emit(self.entry_price)

                except Exception as e:
                    self.error_signal.emit(f"Worker error: {str(e)}\n{traceback.format_exc()}")
                time.sleep(0.1)

            self.data_handler.stop()
            self.predictor.save_models()
            logging.info("Worker stopped")
        except Exception as e:
            self.error_signal.emit(f"Worker crashed: {str(e)}\n{traceback.format_exc()}")

    def train_manual(self, epochs):
        self.predictor.train_manual(epochs, progress_callback=self.training_progress.emit)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NNTicks")
        self.setGeometry(0, 0, 1200, 800)

        self.dark_style = """
            QMainWindow { background-color: #1e1e1e; }
            QLabel { color: #00ff00; font-size: 12px; padding: 5px; }
            QProgressBar { background-color: #333333; border: 1px solid #00ff00; height: 15px; }
            QProgressBar::chunk { background-color: #00ff00; }
            QTabWidget::pane { background-color: #2e2e2e; border: 1px solid #444444; padding: 5px; }
            QTabBar::tab { background-color: #2e2e2e; color: #00ff00; padding: 5px 10px; }
            QTabBar::tab:selected { background-color: #444444; }
            QComboBox, QPushButton { background-color: #333333; color: #00ff00; border: 1px solid #00ff00; padding: 3px 8px; }
            QComboBox:hover, QPushButton:hover { background-color: #ffffff; color: #000000; }
            QTextEdit { background-color: #2e2e2e; color: #00ff00; border: 1px solid #444444; padding: 5px; }
            QLineEdit { background-color: #2e2e2e; color: #00ff00; border: 1px solid #00ff00; padding: 3px; }
            QSpinBox, QDoubleSpinBox { background-color: #2e2e2e; color: #00ff00; border: 1px solid #00ff00; padding: 3px; }
        """
        self.light_style = """
            QMainWindow { background-color: #f0f0f0; }
            QLabel { color: #000000; font-size: 12px; padding: 5px; }
            QProgressBar { background-color: #d0d0d0; border: 1px solid #000000; height: 15px; }
            QProgressBar::chunk { background-color: #00cc00; }
            QTabWidget::pane { background-color: #e0e0e0; border: 1px solid #888888; padding: 5px; }
            QTabBar::tab { background-color: #e0e0e0; color: #000000; padding: 5px 10px; }
            QTabBar::tab:selected { background-color: #c0c0c0; }
            QComboBox, QPushButton { background-color: #d0d0d0; color: #000000; border: 1px solid #000000; padding: 3px 8px; }
            QComboBox:hover, QPushButton:hover { background-color: #ffffff; color: #000000; }
            QTextEdit { background-color: #e0e0e0; color: #000000; border: 1px solid #888888; padding: 5px; }
            QLineEdit { background-color: #e0e0e0; color: #000000; border: 1px solid #000000; padding: 3px; }
            QSpinBox, QDoubleSpinBox { background-color: #e0e0e0; color: #000000; border: 1px solid #000000; padding: 3px; }
        """
        self.setStyleSheet(self.dark_style)

        self.dark_mpf_style = mpf.make_mpf_style(base_mpf_style='charles', rc={'axes.labelcolor':'#00ff00', 'xtick.color':'#00ff00', 'ytick.color':'#00ff00', 'axes.edgecolor':'#444444', 'figure.facecolor':'#1e1e1e', 'axes.facecolor':'#2e2e2e'})
        self.light_mpf_style = mpf.make_mpf_style(base_mpf_style='charles', rc={'axes.labelcolor':'#000000', 'xtick.color':'#000000', 'ytick.color':'#000000', 'axes.edgecolor':'#888888', 'figure.facecolor':'#f0f0f0', 'axes.facecolor':'#e0e0e0'})

        self.theme_selector = QComboBox()
        self.theme_selector.addItems(["Dark", "Light"])
        self.theme_selector.setCurrentText("Dark")

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)

        # Status Bar
        status_layout = QHBoxLayout()
        status_layout.setSpacing(10)
        self.progress = QProgressBar()
        self.progress.setMaximum(10)
        self.status_label = QLabel("Disconnected")
        self.status_label.setFixedWidth(300)
        self.tick_label = QLabel("Current Tick: <span style='color: #ffffff'>N/A</span>")
        self.tick_label.setFixedWidth(150)
        self.mode_label = QLabel(f"Mode: {PREDICTION_MODE}")
        self.mode_label.setFixedWidth(100)
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setFixedWidth(80)
        self.connect_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.connect_btn.clicked.connect(self.toggle_connect)
        self.start_glow_animation()
        status_layout.addWidget(self.progress)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.tick_label)
        status_layout.addWidget(self.mode_label)
        status_layout.addWidget(self.connect_btn)
        layout.addLayout(status_layout)

        # Tabs
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Overview Tab (Tab 0)
        overview_tab = QWidget()
        overview_layout = QVBoxLayout(overview_tab)
        overview_layout.setContentsMargins(5, 5, 5, 5)
        self.overview_fig, self.overview_ax = plt.subplots(figsize=(10, 4))
        self.overview_fig.patch.set_facecolor("#1e1e1e")
        self.overview_canvas = FigureCanvas(self.overview_fig)
        overview_layout.addWidget(self.overview_canvas)
        self.overview_ax.set_facecolor("#2e2e2e")
        self.overview_ax.grid(True, color="#444444")
        self.overview_ax.tick_params(colors="#00ff00", labelrotation=0)
        self.overview_ax.set_xlabel("Time (UTC)", color="#00ff00", fontsize=10)
        self.overview_ax.set_ylabel("Price (USD)", color="#00ff00", fontsize=10)
        self.current_price_line = self.overview_ax.axhline(y=0, color="yellow", linestyle="-", linewidth=1, zorder=10)
        self.current_price_text = self.overview_ax.text(0, 0, "", color="yellow", fontsize=10, ha="right", va="bottom")
        self.pred_label = QLabel("Prediction: N/A")
        self.stats_label = QLabel("Wins: 0 | Losses: 0 | Win Rate: 0%")
        self.level_label = QLabel("Level: 0 | Points: 0")
        self.streak_label = QLabel("Current Streak: 0")
        self.profit_label = QLabel("Profit: 0")
        self.time_since_last_trade_label = QLabel("Time since last trade: N/A")
        self.recent_trades = QTextEdit()
        self.recent_trades.setReadOnly(True)
        self.recent_trades.setFixedHeight(100)
        self.performance_label = QLabel("Avg Confidence: N/A | Trades/Hour: N/A")
        overview_layout.addWidget(self.pred_label)
        overview_layout.addWidget(self.stats_label)
        overview_layout.addWidget(self.level_label)
        overview_layout.addWidget(self.streak_label)
        overview_layout.addWidget(self.profit_label)
        overview_layout.addWidget(self.time_since_last_trade_label)
        overview_layout.addWidget(QLabel("Recent Trades:"))
        overview_layout.addWidget(self.recent_trades)
        overview_layout.addWidget(self.performance_label)
        self.tabs.addTab(overview_tab, "Overview")

        # Rise/Fall Tab (Tab 1)
        rise_fall_tab = QWidget()
        rise_fall_layout = QVBoxLayout(rise_fall_tab)
        rise_fall_layout.setContentsMargins(5, 5, 5, 5)
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.fig.patch.set_facecolor("#1e1e1e")
        self.canvas = FigureCanvas(self.fig)
        rise_fall_layout.addWidget(self.canvas)
        self.line, = self.ax.plot([], [], color="#00ff00", label="Actual")
        self.pred_line, = self.ax.plot([], [], color="#ff00ff", label="Predicted")
        self.ax.set_title("Prediction vs Actual", color="#00ff00", fontsize=10)
        self.ax.set_xlabel("Time (UTC)", color="#00ff00", fontsize=10)
        self.ax.set_ylabel("Price (USD)", color="#00ff00", fontsize=10)
        self.ax.set_facecolor("#2e2e2e")
        self.ax.tick_params(colors="#00ff00")
        self.ax.grid(True, color="#444444")
        self.ax.legend(facecolor="#2e2e2e", edgecolor="#00ff00", labelcolor="#00ff00")
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)
        controls_layout.addWidget(QLabel("Tick Count:"))
        self.tick_selector = QComboBox()
        self.tick_selector.addItems(["1", "2", "3", "4", "5"])
        self.tick_selector.setCurrentText("3")
        self.tick_selector.setFixedWidth(60)
        self.tick_selector.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.tick_selector.currentTextChanged.connect(self.update_tick_count)
        controls_layout.addWidget(self.tick_selector)
        controls_layout.addWidget(QLabel("Mode:"))
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Frequent", "Strict", "Custom"])
        self.mode_selector.setCurrentText(PREDICTION_MODE)
        self.mode_selector.setFixedWidth(100)
        self.mode_selector.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.mode_selector.currentTextChanged.connect(self.update_mode)
        controls_layout.addWidget(self.mode_selector)
        self.custom_threshold_spin = QDoubleSpinBox()
        self.custom_threshold_spin.setRange(0.0, 1.0)
        self.custom_threshold_spin.setSingleStep(0.01)
        self.custom_threshold_spin.setValue(0.75)
        self.custom_threshold_spin.setEnabled(False)
        self.custom_threshold_spin.setFixedWidth(60)
        self.custom_threshold_spin.valueChanged.connect(self.update_custom_threshold)
        controls_layout.addWidget(QLabel("Custom Threshold:"))
        controls_layout.addWidget(self.custom_threshold_spin)
        controls_layout.addStretch()
        rise_fall_layout.addLayout(controls_layout)
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        self.history_text.setFixedHeight(100)
        rise_fall_layout.addWidget(QLabel("Trade History:"))
        rise_fall_layout.addWidget(self.history_text)
        self.export_btn = QPushButton("Export to PDF")
        self.export_btn.setFixedWidth(120)
        self.export_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.export_btn.clicked.connect(self.export_pdf)
        rise_fall_layout.addWidget(self.export_btn)
        self.tabs.addTab(rise_fall_tab, "Rise/Fall")

        # Backtest Tab (Tab 2)
        backtest_tab = QWidget()
        backtest_layout = QVBoxLayout(backtest_tab)
        backtest_layout.setContentsMargins(5, 5, 5, 5)
        self.backtest_file_btn = QPushButton("Select CSV File")
        self.backtest_file_btn.setFixedWidth(120)
        self.backtest_file_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.backtest_file_btn.clicked.connect(self.select_backtest_file)
        backtest_layout.addWidget(self.backtest_file_btn)
        self.backtest_stake_spin = QSpinBox()
        self.backtest_stake_spin.setRange(1, 1000)
        self.backtest_stake_spin.setValue(10)
        self.backtest_stake_spin.setFixedWidth(60)
        backtest_layout.addWidget(QLabel("Stake per Trade:"))
        backtest_layout.addWidget(self.backtest_stake_spin)
        self.backtest_run_btn = QPushButton("Run Backtest")
        self.backtest_run_btn.setFixedWidth(120)
        self.backtest_run_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.backtest_run_btn.clicked.connect(self.run_backtest)
        backtest_layout.addWidget(self.backtest_run_btn)
        self.backtest_results = QTextEdit()
        self.backtest_results.setReadOnly(True)
        self.backtest_results.setFixedHeight(200)
        backtest_layout.addWidget(QLabel("Backtest Results:"))
        backtest_layout.addWidget(self.backtest_results)
        self.tabs.addTab(backtest_tab, "Backtest")

        # Training Tab (Tab 3)
        training_tab = QWidget()
        training_layout = QVBoxLayout(training_tab)
        training_layout.setContentsMargins(5, 5, 5, 5)
        training_layout.addWidget(QLabel("Train Epochs:"))
        self.epoch_selector = QComboBox()
        self.epoch_selector.addItems([str(i) for i in range(500, 10001, 500)])
        self.epoch_selector.setFixedWidth(100)
        self.epoch_selector.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        training_layout.addWidget(self.epoch_selector)
        self.train_btn = QPushButton("Train NN")
        self.train_btn.setFixedWidth(100)
        self.train_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.train_btn.clicked.connect(self.train_nn)
        training_layout.addWidget(self.train_btn)
        self.auto_train_checkbox = QCheckBox("Auto Train After Trade")
        self.auto_train_checkbox.setStyleSheet("color: #00ff00;")
        self.auto_train_checkbox.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        training_layout.addWidget(self.auto_train_checkbox)
        self.train_progress = QProgressBar()
        self.train_progress.setMaximum(100)
        training_layout.addWidget(self.train_progress)
        self.training_status = QLabel("Training Status: Idle")
        self.training_progress_bar = QProgressBar()
        self.training_progress_bar.setMaximum(1000)
        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setFixedHeight(150)
        self.mission_label = QLabel(f"Mission: Complete 5 trades to unlock training (Level 0)")
        self.level_progress = QProgressBar()
        self.level_progress.setMaximum(100)
        self.thoughts_label = QLabel("Bot Thoughts: N/A")
        self.train_status_label = QLabel("Complete 5 trades to enable training")
        self.training_indicator = QLabel("Training: Inactive")
        self.training_indicator.setStyleSheet("color: #ff0000;")
        self.prediction_log = QTextEdit()
        self.prediction_log.setReadOnly(True)
        self.prediction_log.setFixedHeight(150)
        training_layout.addWidget(self.training_status)
        training_layout.addWidget(self.training_progress_bar)
        training_layout.addWidget(QLabel("Training Log:"))
        training_layout.addWidget(self.training_log)
        training_layout.addWidget(QLabel("Prediction Log:"))
        training_layout.addWidget(self.prediction_log)
        training_layout.addWidget(self.mission_label)
        training_layout.addWidget(self.level_progress)
        training_layout.addWidget(self.thoughts_label)
        training_layout.addWidget(self.train_status_label)
        training_layout.addWidget(self.training_indicator)
        training_layout.addStretch()
        self.tabs.addTab(training_tab, "Training")

        # Settings Tab (Tab 4)
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        settings_layout.setContentsMargins(5, 5, 5, 5)
        settings_sub_tabs = QTabWidget()

        # Appearance Sub-Tab
        appearance_tab = QWidget()
        appearance_layout = QVBoxLayout(appearance_tab)
        appearance_layout.addWidget(QLabel("Theme:"))
        appearance_layout.addWidget(self.theme_selector)
        self.theme_selector.setFixedWidth(100)
        self.theme_selector.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        appearance_layout.addWidget(QLabel("Font Size:"))
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 16)
        self.font_size_spin.setValue(12)
        self.font_size_spin.setFixedWidth(60)
        appearance_layout.addWidget(self.font_size_spin)
        self.mute_checkbox = QCheckBox("Mute Alerts")
        self.mute_checkbox.setStyleSheet("color: #00ff00;")
        self.mute_checkbox.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        appearance_layout.addWidget(self.mute_checkbox)
        self.auto_connect_checkbox = QCheckBox("Auto Connect on Start")
        self.auto_connect_checkbox.setStyleSheet("color: #00ff00;")
        self.auto_connect_checkbox.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        appearance_layout.addWidget(self.auto_connect_checkbox)
        appearance_layout.addStretch()
        settings_sub_tabs.addTab(appearance_tab, "Appearance")

        # Chart Settings Sub-Tab
        chart_tab = QWidget()
        chart_layout = QVBoxLayout(chart_tab)
        chart_layout.addWidget(QLabel("Candlestick Grid Color:"))
        self.grid_color_edit = QLineEdit("#444444")
        self.grid_color_edit.setFixedWidth(100)
        chart_layout.addWidget(self.grid_color_edit)
        chart_layout.addWidget(QLabel("Current Price Line Color:"))
        self.price_line_color_edit = QLineEdit("yellow")
        self.price_line_color_edit.setFixedWidth(100)
        chart_layout.addWidget(self.price_line_color_edit)
        chart_layout.addStretch()
        settings_sub_tabs.addTab(chart_tab, "Charts")

        # Training Settings Sub-Tab
        training_settings_tab = QWidget()
        training_settings_layout = QVBoxLayout(training_settings_tab)
        training_settings_layout.addWidget(QLabel("Minimum Trades:"))
        self.min_trades_spin = QSpinBox()
        self.min_trades_spin.setRange(1, 50)
        self.min_trades_spin.setValue(5)
        self.min_trades_spin.setFixedWidth(60)
        training_settings_layout.addWidget(self.min_trades_spin)
        training_settings_layout.addWidget(QLabel("Learning Rate:"))
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.001, 0.1)
        self.learning_rate_spin.setValue(0.01)
        self.learning_rate_spin.setSingleStep(0.001)
        self.learning_rate_spin.setFixedWidth(60)
        training_settings_layout.addWidget(self.learning_rate_spin)
        training_settings_layout.addStretch()
        settings_sub_tabs.addTab(training_settings_tab, "Training")

        # WebSocket Settings Sub-Tab
        websocket_tab = QWidget()
        websocket_layout = QVBoxLayout(websocket_tab)
        websocket_layout.addWidget(QLabel("WebSocket API URL:"))
        self.api_url_edit = QLineEdit(WS_URL)
        self.api_url_edit.setFixedWidth(200)
        websocket_layout.addWidget(self.api_url_edit)
        websocket_layout.addWidget(QLabel("API Key:"))
        self.api_key_edit = QLineEdit(API_TOKEN)
        self.api_key_edit.setFixedWidth(200)
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        websocket_layout.addWidget(self.api_key_edit)
        self.error_log_btn = QPushButton("Get Error Log")
        self.error_log_btn.setFixedWidth(120)
        self.error_log_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.error_log_btn.clicked.connect(self.show_error_log)
        websocket_layout.addWidget(self.error_log_btn)
        websocket_layout.addStretch()
        settings_sub_tabs.addTab(websocket_tab, "WebSocket")

        # About Us/Policy Sub-Tab
        about_tab = QWidget()
        about_layout = QVBoxLayout(about_tab)
        about_text = QTextEdit()
        about_text.setReadOnly(True)
        about_text.setText(
            "NNTicks\n"
            "Copyright © 2025 Ruel McNeil\n"
            "Developed by Grok (xAI)\n\n"
            "Usage Policy:\n"
            "NNTicks is designed for educational and analytical purposes only. It provides tools to visualize market data and simulate trading strategies using a neural network. This software does not constitute financial advice, and users are solely responsible for any trading decisions made based on its outputs. Trading in financial markets involves significant risk, including the potential loss of principal. Past performance is not indicative of future results. Users should consult with a qualified financial advisor before engaging in live trading. The developers and contributors to NNTicks disclaim any liability for losses incurred through the use of this software. By using NNTicks, you agree to these terms and acknowledge that you use it at your own risk."
        )
        about_layout.addWidget(about_text)
        settings_sub_tabs.addTab(about_tab, "About Us/Policy")

        settings_layout.addWidget(settings_sub_tabs)
        self.apply_settings_btn = QPushButton("Apply Settings")
        self.apply_settings_btn.setFixedWidth(120)
        self.apply_settings_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.apply_settings_btn.clicked.connect(self.apply_settings)
        settings_layout.addWidget(self.apply_settings_btn)
        self.tabs.addTab(settings_tab, "Settings")

        # Footer
        self.footer_label = QLabel("Copyright © 2025 Ruel McNeil | Developed by Grok (xAI)")
        self.footer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.footer_label)

        self.worker = TickWorker(tick_count=3, mode=PREDICTION_MODE, api_url=self.api_url_edit.text(), api_key=self.api_key_edit.text())
        self.worker.tick_received.connect(self.update_charts)
        self.worker.prediction_made.connect(self.on_prediction)
        self.worker.entry_signal.connect(self.on_entry)
        self.worker.countdown_signal.connect(self.on_countdown)
        self.worker.outcome_received.connect(self.on_outcome)
        self.worker.error_signal.connect(self.on_error)
        self.worker.training_progress.connect(self.on_training_progress)
        self.worker.prediction_log.connect(self.on_prediction_log)

        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_value = 0
        self.prices = []
        self.times = []
        self.candle_data = []
        self.pred_times = []
        self.pred_prices = []
        self.entries = []
        self.exits = []
        self.wins = 0
        self.losses = 0
        self.confidences = []
        self.trade_times = []
        self.alert_sound = QSoundEffect()
        self.alert_sound.setSource(QUrl.fromLocalFile(r"C:\Users\ruel.mcneil\Desktop\NNTicks\alert.wav"))
        self.last_price = None
        self.streak = 0
        self.profit = 0
        self.last_trade_time = None

        self.train_btn.setEnabled(False)

        self.load_session()

        self.center_window()

        self.time_since_last_trade_timer = QTimer()
        self.time_since_last_trade_timer.timeout.connect(self.update_time_since_last_trade)
        self.time_since_last_trade_timer.start(1000)  # Update every second

    def start_glow_animation(self):
        self.glow_animation = QPropertyAnimation(self.connect_btn, b"styleSheet")
        self.glow_animation.setDuration(1000)
        self.glow_animation.setLoopCount(-1)
        self.glow_animation.setStartValue("background-color: #333333; color: #00ff00; border: 1px solid #00ff00; padding: 3px 8px;")
        self.glow_animation.setEndValue("background-color: #00ff00; color: #000000; border: 1px solid #00ff00; padding: 3px 8px;")
        self.glow_animation.setEasingCurve(QEasingCurve.Type.InOutSine)
        self.glow_animation.start()

    def stop_glow_animation(self):
        if hasattr(self, 'glow_animation'):
            self.glow_animation.stop()
            self.connect_btn.setStyleSheet("background-color: #333333; color: #00ff00; border: 1px solid #00ff00; padding: 3px 8px;")

    def center_window(self):
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        window_geometry = self.frameGeometry()
        window_geometry.moveCenter(screen_geometry.center())
        self.move(window_geometry.topLeft())

    def update_charts(self, price, timestamp):
        try:
            self.prices.append(price)
            self.times.append(timestamp)
            if len(self.prices) > 500:
                self.prices.pop(0)
                self.times.pop(0)

            if self.last_price is not None:
                color = "#00ff00" if price > self.last_price else "#ff0000" if price < self.last_price else "#ffffff"
                self.tick_label.setText(f"Current Tick: <span style='color: {color}'>{price:.3f}</span>")
            else:
                self.tick_label.setText(f"Current Tick: <span style='color: #ffffff'>{price:.3f}</span>")
            self.last_price = price

            minute_start = int(timestamp // 60 * 60)
            if not self.candle_data or self.candle_data[-1]['time'] != minute_start:
                self.candle_data.append({'time': minute_start, 'open': price, 'high': price, 'low': price, 'close': price})
            else:
                candle = self.candle_data[-1]
                candle['high'] = max(candle['high'], price)
                candle['low'] = min(candle['low'], price)
                candle['close'] = price
            if len(self.candle_data) > 60:
                self.candle_data.pop(0)

            current_tab = self.tabs.currentIndex()
            if current_tab == 0:  # Overview
                self.overview_ax.clear()
                df = pd.DataFrame(self.candle_data)
                df['Date'] = pd.to_datetime(df['time'], unit='s')
                df = df[['Date', 'open', 'high', 'low', 'close']].rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
                style = self.dark_mpf_style if self.theme_selector.currentText() == "Dark" else self.light_mpf_style
                mpf.plot(df.set_index('Date'), type='candle', ax=self.overview_ax, style=style, ylabel='')
                self.current_price_line = self.overview_ax.axhline(y=price, color=self.price_line_color_edit.text(), linestyle="-", linewidth=1, zorder=10)
                self.current_price_text.set_position((df['Date'].iloc[-1], price))
                self.current_price_text.set_text(f"{price:.3f}")
                self.current_price_text.set_y(price + 0.01)
                self.current_price_text.set_color(self.price_line_color_edit.text())
                self.overview_canvas.draw()
            elif current_tab == 1:  # Rise/Fall
                self.ax.clear()
                color_actual = '#00ff00' if self.theme_selector.currentText() == "Dark" else '#00cc00'
                color_pred = '#ff00ff' if self.theme_selector.currentText() == "Dark" else '#ff00cc'
                self.line, = self.ax.plot(self.times, self.prices, color=color_actual, label="Actual")
                if self.pred_times:
                    self.pred_line, = self.ax.plot(self.pred_times, self.pred_prices, color=color_pred, label="Predicted")
                for entry_time, entry_price in self.entries[-5:]:
                    self.ax.plot(entry_time, entry_price, 'go', label="Entry" if len(self.entries) == 1 else "")
                for exit_time, exit_price in self.exits[-5:]:
                    self.ax.plot(exit_time, exit_price, 'ro', label="Exit" if len(self.exits) == 1 else "")
                self.ax.legend(facecolor='#2e2e2e' if self.theme_selector.currentText() == "Dark" else '#e0e0e0',
                              edgecolor='#00ff00' if self.theme_selector.currentText() == "Dark" else '#000000',
                              labelcolor='#00ff00' if self.theme_selector.currentText() == "Dark" else '#000000')
                self.ax.set_facecolor('#2e2e2e' if self.theme_selector.currentText() == "Dark" else '#e0e0e0')
                self.ax.tick_params(colors='#00ff00' if self.theme_selector.currentText() == "Dark" else '#000000')
                self.ax.grid(True, color='#444444' if self.theme_selector.currentText() == "Dark" else '#888888')
                self.canvas.draw()
                mplcursors.cursor(self.line, hover=True)
                if self.pred_line:
                    mplcursors.cursor(self.pred_line, hover=True)

        except Exception as e:
            logging.error(f"Chart update error: {str(e)}\n{traceback.format_exc()}")
            self.status_label.setText("Chart error - check logs")

    def on_prediction(self, direction, ticks, confidence, model, thoughts, pred_price):
        self.status_label.setText(f"Prediction: {direction} in {ticks} ticks [Confidence: {confidence:.3f}] [{model}]")
        self.pred_label.setText(f"Prediction: {direction} [{confidence:.3f}]")
        self.thoughts_label.setText(f"Bot Thoughts: {thoughts}")
        self.countdown_value = 10
        self.progress.setMaximum(10)
        self.progress.setValue(10)
        self.countdown_timer.start(1000)
        if not self.mute_checkbox.isChecked():
            winsound.PlaySound(r"C:\Users\ruel.mcneil\Desktop\NNTicks\alert.wav", winsound.SND_ASYNC)
        self.pred_times = [time.time()]
        self.pred_prices = [pred_price]
        pred_end_price = pred_price + (0.1 if direction == "Rise" else -0.1)
        self.pred_times.append(time.time() + ticks * 0.2)
        self.pred_prices.append(pred_end_price)
        if self.tabs.currentIndex() == 1:
            self.update_charts(pred_price, time.time())
        self.confidences.append(confidence)

    def on_prediction_log(self, log_entry):
        self.prediction_log.append(log_entry)
        if self.prediction_log.document().lineCount() > 100:
            self.prediction_log.setPlainText("\n".join(self.prediction_log.toPlainText().split("\n")[-100:]))

    def update_countdown(self):
        if self.countdown_value > 0:
            self.countdown_value -= 1
            self.progress.setValue(self.countdown_value)
            if self.countdown_value == 0:
                self.countdown_timer.stop()

    def on_entry(self, entry_price):
        self.status_label.setText(f"Enter trade now at {entry_price:.3f}!")
        self.progress.setMaximum(self.worker.tick_count)
        self.progress.setValue(0)
        self.entries.append((time.time(), entry_price))
        if self.tabs.currentIndex() == 1:
            self.update_charts(entry_price, time.time())

    def on_countdown(self, tick_count):
        self.status_label.setText(f"Countdown: {tick_count}/{self.worker.tick_count} ticks")
        self.progress.setValue(tick_count)

    def on_outcome(self, direction, entry_price, end_price, outcome):
        self.status_label.setText(f"Outcome: {direction} from {entry_price:.3f} to {end_price:.3f} - {outcome}")
        self.progress.setValue(0)
        self.exits.append((time.time(), end_price))
        trade_log = f"{direction}: {entry_price:.3f} -> {end_price:.3f} ({outcome})"
        self.history_text.append(trade_log)
        self.recent_trades.append(trade_log)
        stake = 10
        if outcome == "Win":
            self.wins += 1
            self.profit += stake
            self.streak = self.streak + 1 if self.streak > 0 else 1
        else:
            self.losses += 1
            self.profit -= stake
            self.streak = self.streak - 1 if self.streak < 0 else -1
        total = self.wins + self.losses
        win_rate = (self.wins / total * 100) if total > 0 else 0
        self.stats_label.setText(f"Wins: {self.wins} | Losses: {self.losses} | Win Rate: {win_rate:.1f}%")
        self.level_label.setText(f"Level: {self.worker.predictor.nn.level} | Points: {self.worker.predictor.nn.points}")
        self.streak_label.setText(f"Current Streak: {self.streak}")
        self.profit_label.setText(f"Profit: {self.profit}")
        self.last_trade_time = time.time()
        self.trade_times.append(time.time())
        avg_conf = sum(self.confidences[-10:]) / len(self.confidences[-10:]) if self.confidences else 0
        trades_per_hour = len([t for t in self.trade_times if time.time() - t < 3600]) / (3600 / 3600)
        self.performance_label.setText(f"Avg Confidence: {avg_conf:.3f} | Trades/Hour: {trades_per_hour:.1f}")
        if not self.mute_checkbox.isChecked():
            winsound.PlaySound(r"C:\Users\ruel.mcneil\Desktop\NNTicks\alert.wav", winsound.SND_ASYNC)
        self.pred_times = []
        self.pred_prices = []
        if self.tabs.currentIndex() == 1:
            self.update_charts(end_price, time.time())
        trades_left = self.worker.predictor.min_trades - len(self.worker.predictor.history)
        self.mission_label.setText(f"Mission: Complete {max(0, trades_left)} trades to unlock training (Level {self.worker.predictor.nn.level})")
        if trades_left > 0:
            self.train_status_label.setText(f"Complete {trades_left} more trades to enable training")
            self.train_btn.setEnabled(False)
        else:
            self.train_status_label.setText("Training enabled")
            self.train_btn.setEnabled(True)
        self.save_session()

    def update_tick_count(self, value):
        self.worker.set_tick_count(int(value))

    def update_mode(self, value):
        self.worker.set_mode(value)
        self.mode_label.setText(f"Mode: {value}")
        self.custom_threshold_spin.setEnabled(value == "Custom")
        if value == "Custom":
            self.worker.predictor.custom_threshold = self.custom_threshold_spin.value()

    def update_custom_threshold(self, value):
        if self.mode_selector.currentText() == "Custom":
            self.worker.predictor.custom_threshold = value

    def toggle_connect(self):
        self.stop_glow_animation()
        if self.worker.running:
            self.worker.running = False
            self.worker.wait()
            self.connect_btn.setText("Connect")
            self.status_label.setText("Disconnected")
            self.start_glow_animation()
        else:
            self.worker = TickWorker(
                tick_count=int(self.tick_selector.currentText()),
                mode=self.mode_selector.currentText(),
                api_url=self.api_url_edit.text(),
                api_key=self.api_key_edit.text()
            )
            self.worker.predictor.custom_threshold = self.custom_threshold_spin.value()
            self.worker.tick_received.connect(self.update_charts)
            self.worker.prediction_made.connect(self.on_prediction)
            self.worker.entry_signal.connect(self.on_entry)
            self.worker.countdown_signal.connect(self.on_countdown)
            self.worker.outcome_received.connect(self.on_outcome)
            self.worker.error_signal.connect(self.on_error)
            self.worker.training_progress.connect(self.on_training_progress)
            self.worker.prediction_log.connect(self.on_prediction_log)
            self.worker.predictor.auto_train = self.auto_train_checkbox.isChecked()
            self.worker.predictor.nn.learning_rate = self.learning_rate_spin.value()
            self.worker.running = True
            self.worker.start()
            self.connect_btn.setText("Disconnect")
            self.status_label.setText("Connected")
            self.mode_label.setText(f"Mode: {self.worker.mode}")

    def on_error(self, error_msg):
        logging.error(error_msg)
        self.status_label.setText("Error occurred - check logs")
        self.worker.running = False
        self.connect_btn.setText("Connect")
        self.start_glow_animation()

    def train_nn(self):
        try:
            self.training_status.setText("Training Status: Running")
            self.training_indicator.setText("Training: Active")
            self.training_indicator.setStyleSheet("color: #00ff00;")
            self.training_log.clear()
            self.training_progress_bar.setValue(0)
            epochs = int(self.epoch_selector.currentText())
            self.training_progress_bar.setMaximum(epochs)
            self.worker.train_manual(epochs)
            self.level_label.setText(f"Level: {self.worker.predictor.nn.level} | Points: {self.worker.predictor.nn.points}")
            self.train_progress.setValue(self.worker.predictor.nn.points % 100)
            self.level_progress.setValue(self.worker.predictor.nn.points % 100)
            self.mission_label.setText(f"Mission: Complete {max(0, self.worker.predictor.min_trades - len(self.worker.predictor.history))} trades to unlock training (Level {self.worker.predictor.nn.level})")
            logging.info(f"Trained NN for {epochs} epochs. Points: {self.worker.predictor.nn.points}, Level: {self.worker.predictor.nn.level}")
            self.training_status.setText("Training Status: Complete")
            self.training_indicator.setText("Training: Inactive")
            self.training_indicator.setStyleSheet("color: #ff0000;")
            self.save_session()
        except ValueError as e:
            logging.warning(str(e))
            self.status_label.setText(str(e))
            self.training_status.setText("Training Status: Idle")
            self.training_indicator.setText("Training: Inactive")
            self.training_indicator.setStyleSheet("color: #ff0000;")
        except Exception as e:
            logging.error(f"Training error: {str(e)}\n{traceback.format_exc()}")
            self.status_label.setText("Training error - check logs")
            self.training_status.setText("Training Status: Idle")
            self.training_indicator.setText("Training: Inactive")
            self.training_indicator.setStyleSheet("color: #ff0000;")

    def on_training_progress(self, epoch, loss, learned_info):
        self.training_progress_bar.setValue(epoch + 1)
        self.training_log.append(f"Epoch {epoch + 1}: Loss = {loss:.4f}, Learned: {learned_info}")
        self.training_status.setText(f"Training Status: Epoch {epoch + 1}/{self.training_progress_bar.maximum()}")

    def export_pdf(self):
        try:
            pdf_file = r"C:\Users\ruel.mcneil\Desktop\NNTicks\NNTicks_Report.pdf"
            c = canvas.Canvas(pdf_file, pagesize=letter)
            c.setFont("Helvetica", 12)
            c.drawString(100, 750, "NNTicks Trade Report")
            c.drawString(100, 730, f"Wins: {self.wins} | Losses: {self.losses} | Win Rate: {self.stats_label.text().split()[-1]}")
            c.drawString(100, 710, "Trade History:")
            y = 690
            for line in self.history_text.toPlainText().split("\n")[-20:]:
                c.drawString(100, y, line[:80])
                y -= 20
            chart_path = r"C:\Users\ruel.mcneil\Desktop\NNTicks\chart.png"
            self.fig.savefig(chart_path)
            c.drawImage(chart_path, 100, 300, width=400, height=300)
            c.showPage()
            c.save()
            logging.info(f"Exported report to {pdf_file}")
            QMessageBox.information(self, "Export", f"Report exported to {pdf_file}")
        except Exception as e:
            logging.error(f"PDF export error: {str(e)}\n{traceback.format_exc()}")
            self.status_label.setText("PDF export failed - check logs")

    def apply_settings(self):
        font_size = self.font_size_spin.value()
        theme = self.theme_selector.currentText()
        style = self.dark_style if theme == "Dark" else self.light_style
        self.setStyleSheet(style.replace("font-size: 12px", f"font-size: {font_size}px"))
        self.worker.predictor.min_trades = self.min_trades_spin.value()
        self.worker.predictor.nn.learning_rate = self.learning_rate_spin.value()
        self.worker.predictor.auto_train = self.auto_train_checkbox.isChecked()
        self.worker.predictor.custom_threshold = self.custom_threshold_spin.value()
        self.tabs.currentWidget().layout().update()
        self.save_session()
        QMessageBox.information(self, "Settings", "Settings applied successfully. Reconnect to use new WebSocket settings.")

    def show_error_log(self):
        try:
            with open(r"C:\Users\ruel.mcneil\Desktop\NNTicks\nnticks.log", "r") as f:
                logs = f.read()
            QMessageBox.information(self, "Error Log", logs[-2000:])
        except Exception as e:
            QMessageBox.warning(self, "Error Log", f"Failed to retrieve log: {str(e)}")

    def save_session(self):
        session_data = {
            "settings": {
                "theme": self.theme_selector.currentText(),
                "font_size": self.font_size_spin.value(),
                "mute_alerts": self.mute_checkbox.isChecked(),
                "auto_connect": self.auto_connect_checkbox.isChecked(),
                "grid_color": self.grid_color_edit.text(),
                "price_line_color": self.price_line_color_edit.text(),
                "min_trades": self.min_trades_spin.value(),
                "learning_rate": self.learning_rate_spin.value(),
                "api_url": self.api_url_edit.text(),
                "api_key": self.api_key_edit.text(),
                "tick_count": self.tick_selector.currentText(),
                "mode": self.mode_selector.currentText(),
                "custom_threshold": self.custom_threshold_spin.value()
            },
            "stats": {
                "wins": self.wins,
                "losses": self.losses,
                "history": self.history_text.toPlainText(),
                "recent_trades": self.recent_trades.toPlainText(),
                "profit": self.profit,
                "streak": self.streak,
                "last_trade_time": self.last_trade_time
            }
        }
        with open(r"C:\Users\ruel.mcneil\Desktop\NNTicks\session.json", "w") as f:
            json.dump(session_data, f)
        self.worker.predictor.save_models()

    def load_session(self):
        try:
            with open(r"C:\Users\ruel.mcneil\Desktop\NNTicks\session.json", "r") as f:
                session_data = json.load(f)
            settings = session_data.get("settings", {})
            self.theme_selector.setCurrentText(settings.get("theme", "Dark"))
            self.font_size_spin.setValue(settings.get("font_size", 12))
            self.mute_checkbox.setChecked(settings.get("mute_alerts", False))
            self.auto_connect_checkbox.setChecked(settings.get("auto_connect", False))
            self.grid_color_edit.setText(settings.get("grid_color", "#444444"))
            self.price_line_color_edit.setText(settings.get("price_line_color", "yellow"))
            self.min_trades_spin.setValue(settings.get("min_trades", 5))
            self.learning_rate_spin.setValue(settings.get("learning_rate", 0.01))
            self.api_url_edit.setText(settings.get("api_url", WS_URL))
            self.api_key_edit.setText(settings.get("api_key", API_TOKEN))
            self.tick_selector.setCurrentText(settings.get("tick_count", "3"))
            self.mode_selector.setCurrentText(settings.get("mode", PREDICTION_MODE))
            self.custom_threshold_spin.setValue(settings.get("custom_threshold", 0.75))

            stats = session_data.get("stats", {})
            self.wins = stats.get("wins", 0)
            self.losses = stats.get("losses", 0)
            self.history_text.setText(stats.get("history", ""))
            self.recent_trades.setText(stats.get("recent_trades", ""))
            self.profit = stats.get("profit", 0)
            self.streak = stats.get("streak", 0)
            self.last_trade_time = stats.get("last_trade_time", None)
            total = self.wins + self.losses
            win_rate = (self.wins / total * 100) if total > 0 else 0
            self.stats_label.setText(f"Wins: {self.wins} | Losses: {self.losses} | Win Rate: {win_rate:.1f}%")
            self.streak_label.setText(f"Current Streak: {self.streak}")
            self.profit_label.setText(f"Profit: {self.profit}")

            self.worker.predictor.load_models()
            self.level_label.setText(f"Level: {self.worker.predictor.nn.level} | Points: {self.worker.predictor.nn.points}")
            self.train_progress.setValue(self.worker.predictor.nn.points % 100)
            self.level_progress.setValue(self.worker.predictor.nn.points % 100)
            trades_left = self.worker.predictor.min_trades - len(self.worker.predictor.history)
            self.mission_label.setText(f"Mission: Complete {max(0, trades_left)} trades to unlock training (Level {self.worker.predictor.nn.level})")
            if trades_left > 0:
                self.train_status_label.setText(f"Complete {trades_left} more trades to enable training")
                self.train_btn.setEnabled(False)
            else:
                self.train_status_label.setText("Training enabled")
                self.train_btn.setEnabled(True)

            style = self.dark_style if self.theme_selector.currentText() == "Dark" else self.light_style
            self.setStyleSheet(style.replace("font-size: 12px", f"font-size: {self.font_size_spin.value()}px"))
            self.mode_label.setText(f"Mode: {self.mode_selector.currentText()}")
        except FileNotFoundError:
            logging.info("No session file found, starting fresh.")
            self.mode_label.setText(f"Mode: {PREDICTION_MODE}")
        except Exception as e:
            logging.error(f"Session load error: {str(e)}\n{traceback.format_exc()}")

    def closeEvent(self, event):
        reply = QMessageBox.question(self, "Exit", "Are you sure you want to exit NNTicks?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.worker.running = False
            self.worker.wait()
            self.countdown_timer.stop()
            self.time_since_last_trade_timer.stop()
            self.save_session()
            self.worker.deleteLater()
            event.accept()
        else:
            event.ignore()

    def select_backtest_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
        if file_name:
            self.backtest_file = file_name
            self.backtest_file_btn.setText(os.path.basename(file_name))

    def run_backtest(self):
        if not hasattr(self, 'backtest_file'):
            QMessageBox.warning(self, "Backtest", "Please select a CSV file first.")
            return
        stake = self.backtest_stake_spin.value()
        try:
            wins, losses, win_rate = self.worker.predictor.backtest(self.backtest_file, tick_count=int(self.tick_selector.currentText()), stake=stake, mode=self.mode_selector.currentText())
            self.backtest_results.setText(f"Wins: {wins}\nLosses: {losses}\nWin Rate: {win_rate:.2f}%")
        except Exception as e:
            logging.error(f"Backtest error: {str(e)}\n{traceback.format_exc()}")
            QMessageBox.warning(self, "Backtest", f"Error during backtest: {str(e)}")

    def update_time_since_last_trade(self):
        if self.last_trade_time:
            time_since = int(time.time() - self.last_trade_time)
            self.time_since_last_trade_label.setText(f"Time since last trade: {time_since} seconds")
        else:
            self.time_since_last_trade_label.setText("Time since last trade: N/A")

if __name__ == "__main__":
    app = QApplication(sys.argv)

    splash = QSplashScreen(QPixmap(300, 100))
    splash.setStyleSheet("background-color: #1e1e1e;")
    loading_label = QLabel("Loading NNTicks...", splash)
    loading_label.setStyleSheet("color: #00ff00; font-size: 14px;")
    loading_label.move(90, 20)
    loading_bar = QProgressBar(splash)
    loading_bar.setGeometry(50, 60, 200, 20)
    loading_bar.setStyleSheet("QProgressBar { border: 1px solid #00ff00; background-color: #333333; } QProgressBar::chunk { background-color: #00ff00; }")
    loading_bar.setMaximum(100)
    splash.show()

    window = MainWindow()
    for i in range(101):
        loading_bar.setValue(i)
        QApplication.processEvents()
        time.sleep(0.02)

    window.show()
    splash.finish(window)
    if window.auto_connect_checkbox.isChecked():
        window.toggle_connect()
    sys.exit(app.exec())