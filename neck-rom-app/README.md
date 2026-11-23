# Webcam Neck Mobility Check for Older Adults

**頸部活動度自我檢測**：本程式只使用一般筆電的網路攝影機，透過 MediaPipe 臉部與姿勢偵測估算頭部角度，引導長者完成六個頸部動作（前屈、後仰、左右側彎、左右轉），並測量頸椎活動度（CROM）。系統同時偵測肩膀，若長者以轉動身體來代償，會即時提醒「請保持肩膀不動」，並在結果中分別顯示校正前與校正後的角度。

> ⚠️ **非醫療診斷設備 / Not a medical diagnostic device.** For self-monitoring of neck mobility only. Stop immediately if it hurts.

This program uses an ordinary laptop webcam to measure cervical range of motion (CROM). It estimates head pose with MediaPipe, guides the user through six neck movements (flexion, extension, left/right lateral flexion, left/right rotation) and reports the maximum angle of each one. It also tracks the shoulders. If the user turns their trunk to compensate (common in older adults with a stiff neck), the program warns them in real time ("請保持肩膀不動") and reports both the uncorrected and the trunk-corrected neck angle.

---

## 1. Installation

Python **3.10–3.12** is the safest choice for mediapipe (tested with 3.11 + mediapipe 1.2.3 on Windows 11).

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt pytest
python download_models.py        # downloads face_landmarker.task + pose_landmarker_lite.task into models/
```

macOS: the first time you run it, grant Camera permission to Terminal / VS Code (System Settings → Privacy & Security → Camera).

## 2. Running

```bash
python run.py                              # webcam 0, Traditional Chinese UI
python run.py --user grandma               # one name per older adult -> data/grandma/
python run.py --camera 1                   # another webcam
python run.py --lang en                    # English UI
python run.py --log-csv out/log.csv        # per-frame log (for comparison with a phone IMU)
python run.py --save-screens out/          # save a screenshot of each step + the result screen (for slides)
```

Other options: `--video file.mp4` (use a video instead of the webcam), `--auto` (start automatically after 1 s), `--headless` (no window), `--no-pose` (disable trunk compensation), `--no-mirror` (do not mirror the display), `--font path.ttc` (use a different CJK font).

### Keys

| Key | Action |
|---|---|
| SPACE | Start / next step / end the measurement early / back to home (on the result screen) |
| S | Skip the current movement |
| C | Recalibrate the neutral pose (then continue with the current movement) |
| R | Measure again from the start |
| H | History (on the home / result screen) |
| L | Switch Chinese ↔ English |
| F | Fullscreen |
| Q / ESC | Quit |

The most common actions are also on-screen buttons you can click with the mouse: **EN / 中文** (top-right, every screen), **開始 Start** and **歷史紀錄 History** (home), **下一步 Next** and **跳過 Skip** (during the measurement), and **重新檢測 Again / 回首頁 Home / 歷史紀錄 History** (result screen). On the history screen, click anywhere to go back.

### Measurement flow

Calibration: look straight ahead for a 3 s preparation and then hold still for 2 s. If the head moves more than 3° or the face is lost, calibration restarts automatically. Then, for each movement: prepare 3 s → measure 7 s (the maximum angle is kept) → return to center 2.5 s. Results are saved to `data/<user>/<timestamp>.json` with `rom_head` (filtered head angle), `rom_head_raw`, `rom_neck` (trunk subtracted), `max_trunk` (trunk actually subtracted), `max_trunk_raw` (Pose angle before the guards), `compensation`, the number of valid frames and `rejected` (rejected frames per reason). `rom` is `rom_neck` only when compensation was detected, otherwise `rom_head`.

Trunk compensation guards (MediaPipe Pose partly infers shoulder depth from the head, so a head-only turn can make the shoulders appear to move 15–30°): trunk angles below 8° are ignored; for rotation, only as much trunk turn is counted as the 2D shoulder width narrowing supports (`acos(width / width_at_calibration)`, ignored if the width shrank less than 2%); and never more than half of the head angle is subtracted. "Move closer" is only a hint (based on forehead-to-chin face height); it does not reject frames.

## 3. Checking the angle signs (do this first with a real webcam)

On the home screen, the top-right corner shows `yaw / pitch / roll` and `肩膀 yaw / roll` (shoulders). Check after calibrating (or directly on the home screen, using the absolute values):

- turn the head to the subject's **own left** → **yaw positive**
- look **up** → **pitch positive**
- **left ear towards left shoulder** → **roll positive**
- **left shoulder moves back** → shoulder yaw positive; **lean the trunk to the left** → shoulder roll positive

If an axis has the wrong sign, flip the sign of that axis only, in `angles_from_matrix` (or `trunk_angles`) in **`neck_rom/geometry.py`**. Do not change it anywhere else.

**Automated checks done (2026-11-22):**
- `pytest`: 24 tests pass. They cover the angle ↔ matrix round trip, the physical meaning of every sign with synthetic landmarks, calibration while sitting 15° off-axis, the shoulder signs, and the 6-movement flow with synthetic data including trunk compensation.
- Synthetic video (`tools/make_test_video.py`, `portrait.jpg` rotated ±25° in the image plane): with `--no-pose`, tilt left = 24.6° and tilt right = 25.0°, so **the head roll sign is correct**. With Pose enabled the shoulders rotate together with the image, and the app reports `compensation=True` with about 17° of trunk movement, so **the trunk_roll sign is correct too**.
- **Not yet checked:** the yaw/pitch signs with a real person in front of the webcam, and the stability of `trunk_yaw` (it fluctuates by about ±3–4° on a still video).

## 4. Tests

```bash
pytest -q                                            # no webcam needed
python tools/make_test_video.py                      # creates test.mp4 (84 s)
python run.py --video test.mp4 --auto --headless --save-screens out/ --user test
python run.py --video test.mp4 --auto --headless --no-pose --user test
```

The video runs print a results table in the terminal. Use `--user test` so test runs do not mix with a real person's history. The team should also **record its own video of about 1.5 minutes** doing all 6 movements (run it with `--auto`), both for repeatable testing and as a backup video for the demo.

## 5. Project layout

```
neck_rom/geometry.py   angle convention, head matrix from landmarks, shoulder angles  (the ONLY place to fix signs)
neck_rom/filters.py    One-Euro filter
neck_rom/tracker.py    MediaPipe Face + Pose Landmarker (Tasks API, VIDEO mode)
neck_rom/session.py    calibration, filtering, outlier rejection, 6-movement flow, ROM, trunk compensation
neck_rom/i18n.py       all UI strings (zh default, en)
neck_rom/ui.py         drawing with OpenCV + Pillow (CJK font): live / result / history screens
neck_rom/app.py        main loop, CLI, keys, JSON saving, CSV log
neck_rom/storage.py    save / load sessions
tools/make_test_video.py  builds a test video from portrait.jpg
```

## 6. Limitations

- **Not a diagnostic device.** It is only meant for tracking trends yourself.
- **Flexion / extension are not corrected for trunk movement** (that needs the hips in view, which a laptop webcam usually cannot see). This is future work.
- `trunk_yaw` comes from `pose_world_landmarks` (depth estimated from a 2D image), so it is noisier than `trunk_roll`.
- The head angle is built from a few landmarks (eyes, forehead, chin), not from a full 3D head model. Beyond about 60–70° of rotation the landmarks can become less accurate.
- The **reference values** (50/60/45/45/80/80° in `session.py`) are typical values for healthy adults; older adults are usually lower. **Cite a specific source** (e.g. the AMA Guides, or an age-stratified review) and adjust the numbers if needed before putting them on slides.
- Small test group (2–3 older adults), not enough for statistical conclusions.
- **Do not commit** the older adults' `data/` to a public repo (`.gitignore` already excludes it).

Original paper: Chotikkakamthorn et al., *Computers in Biology and Medicine* 195, 110620 (2025). The paper's model is used for offline accuracy comparison; the demo uses MediaPipe because it runs in real time on an ordinary laptop CPU.
