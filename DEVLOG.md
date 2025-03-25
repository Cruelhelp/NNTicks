# 📓 NNTicks Development Log

Welcome to the official development log for **NNTicks** — a neural-network-powered tick prediction desktop app. This document tracks feature progress, improvements, known issues, and development goals.

---

## 🚀 Current Status
> NNTicks is in **active development**. Core functionality is in place, and UI improvements, performance tweaks, and advanced features are ongoing.

---

## ✅ Completed Features

- [x] Live WebSocket price fetching from Deriv
- [x] Basic neural network implementation
- [x] Prediction confidence threshold logic
- [x] Save/load trained model to `model.pkl`
- [x] Basic GUI using PyQt6
- [x] Manual Rise/Fall input system
- [x] Chart plotting (tick & candlestick modes)
- [x] Export trade logs to PDF
- [x] Dark theme support
- [x] README + Screenshots + .gitignore

---

## 🛠 In Progress

- [ ] GUI improvements (responsive layout, clearer labels)
- [ ] Auto-training toggle in GUI
- [ ] Tooltip/context display when cursor in prediction box
- [ ] Trade history log with better formatting
- [ ] Visual alert when prediction confidence is high
- [ ] Model training progress bar
- [ ] Export training history to CSV
- [ ] Clean up directory structure

---

## 🐞 Known Issues / Bugs

- [ ] Occasional lag when model is training while fetching live data
- [ ] Tooltip doesn’t always disappear after focus shift
- [ ] Model may overfit if trained repeatedly on same inputs

---

## 🔮 Upcoming Ideas

- [ ] Switchable models (RNN, LSTM later?)
- [ ] Web dashboard or remote monitoring
- [ ] GUI themes (light/dark toggle)
- [ ] Trade simulation and result tracking
- [ ] Real-time prediction visualization (animated ticks)

---

## 🧠 Notes

- Training is more accurate with ≥30 ticks
- Confidence threshold is adjustable in `config.py` or GUI
- Trade export is located in local working dir as PDF

---

## 🕓 Timeline

| Date       | Update                                  |
|------------|------------------------------------------|
| 2024-03-23 | Repo cleanup, added .gitignore, screenshots |
| 2024-03-24 | Added README badges and dev log          |
| 2024-03-25 | Tooltip/cursor context planned            |
| ...        | (Add new entries here as you develop)     |

---

Happy building!  
**— Ruel McNeil**
