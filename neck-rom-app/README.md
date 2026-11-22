# Webcam Neck Mobility Check for Older Adults

**頸部活動度自我檢測**：本程式只使用一般筆電的網路攝影機，透過 MediaPipe 臉部與姿勢偵測估算頭部角度，引導長者完成六個頸部動作（前屈、後仰、左右側彎、左右轉），並測量頸椎活動度（CROM）。系統同時偵測肩膀，若長者以轉動身體來代償，會即時提醒「請保持肩膀不動」，並在結果中分別顯示校正前與校正後的角度。

> ⚠️ **非醫療診斷設備**：本系統僅供頸部活動度自我追蹤參考。如有疼痛請立即停止。

---

## 1. Cài đặt

Python **3.10–3.12** là an toàn nhất cho mediapipe (đã chạy với 3.11 + mediapipe 1.2.3 trên Windows 11).

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt pytest
python download_models.py        # tải face_landmarker.task + pose_landmarker_lite.task vào models/
```

macOS: lần đầu chạy phải cấp quyền Camera cho Terminal / VS Code (System Settings → Privacy & Security → Camera).

## 2. Chạy

```bash
python run.py                              # webcam 0, tiếng Trung
python run.py --user ba_noi                # mỗi người lớn tuổi một tên -> data/ba_noi/
python run.py --camera 1                   # webcam khác
python run.py --lang en                    # giao diện tiếng Anh
python run.py --log-csv out/log.csv        # log từng frame (để so với IMU điện thoại)
python run.py --save-screens out/          # lưu ảnh từng bước + màn kết quả (làm slide)
```

Các tham số khác: `--video file.mp4` (thay webcam), `--auto` (tự bắt đầu sau 1 giây), `--headless` (không mở cửa sổ), `--no-pose` (tắt bù thân), `--no-mirror` (không lật gương), `--font path.ttc` (font CJK khác).

### Phím tắt

| Phím | Chức năng |
|---|---|
| SPACE | Bắt đầu / bước tiếp / kết thúc đo sớm / về màn home (ở màn kết quả) |
| S | Bỏ qua động tác hiện tại |
| C | Hiệu chỉnh lại tư thế trung tính (rồi làm tiếp động tác hiện tại) |
| R | Đo lại từ đầu |
| H | Lịch sử (ở màn home / kết quả) |
| L | Đổi Trung ↔ Anh |
| F | Toàn màn hình |
| Q / ESC | Thoát |

### Luồng đo

Hiệu chỉnh: nhìn thẳng 3 giây chuẩn bị + 2 giây đứng yên (đầu rung > 3° hoặc mất mặt thì tự hiệu chỉnh lại). Sau đó mỗi động tác: chuẩn bị 3s → đo 7s (lấy góc lớn nhất) → về giữa 2.5s. Kết quả lưu ở `data/<user>/<thời gian>.json`, gồm `rom_head` (góc đầu sau lọc), `rom_head_raw`, `rom_neck` (đã trừ thân), `max_trunk`, `compensation`, số frame hợp lệ.

## 3. Kiểm tra dấu góc (việc đầu tiên khi chạy webcam thật)

Ở màn home, góc phải trên hiện `yaw / pitch / roll` và `肩膀 yaw / roll`. Sau khi hiệu chỉnh (hoặc ngay ở home, giá trị tuyệt đối):

- quay đầu sang **trái thật** (của người được đo) → **yaw dương**
- **ngửa** lên → **pitch dương**
- **tai trái về vai trái** → **roll dương**
- xoay **vai trái ra sau** → `肩膀` yaw dương; **nghiêng thân sang trái** → `肩膀` roll dương

Sai trục nào thì đổi dấu đúng trục đó trong `angles_from_matrix` (hoặc `trunk_angles`) ở **`neck_rom/geometry.py`**, không sửa chỗ khác.

**Đã kiểm tra tự động (22/11/2026):**
- `pytest`: 24 test qua (khứ hồi góc ↔ ma trận, ý nghĩa vật lý của từng dấu với landmark giả, hiệu chỉnh khi ngồi lệch 15°, dấu vai, luồng 6 động tác với dữ liệu giả có bù thân).
- Video tổng hợp (`tools/make_test_video.py`, ảnh `portrait.jpg` xoay ±25° trong mặt phẳng ảnh): `--no-pose` cho tilt trái 24.6°, tilt phải 25.0°, nên **dấu roll của đầu đúng**. Có Pose thì vai xoay cùng ảnh, app báo `compensation=True`, thân khoảng 17°, nên **dấu trunk_roll cũng đúng**.
- **Chưa kiểm tra:** dấu yaw/pitch với người thật trước webcam, và độ ổn định của `trunk_yaw` (trên video tĩnh nó dao động khoảng ±3–4°).

## 4. Test

```bash
pytest -q                                            # không cần webcam
python tools/make_test_video.py                      # tạo test.mp4 (84 s)
python run.py --video test.mp4 --auto --headless --save-screens out/ --user test
python run.py --video test.mp4 --auto --headless --no-pose --user test
```

Lệnh chạy video in bảng kết quả ra terminal. Nên dùng `--user test` để không lẫn vào lịch sử của người thật. Ngoài ra cả nhóm nên **tự quay một video khoảng 1 phút 30 giây** làm đủ 6 động tác (chạy cùng `--auto`) để test lặp lại và làm video dự phòng cho demo.

## 5. Cấu trúc

```
neck_rom/geometry.py   quy ước góc, ma trận đầu từ landmark, góc vai   (chỗ DUY NHẤT để sửa dấu)
neck_rom/filters.py    One-Euro filter
neck_rom/tracker.py    MediaPipe Face + Pose Landmarker (Tasks API, VIDEO mode)
neck_rom/session.py    hiệu chỉnh, lọc, loại outlier, luồng 6 động tác, ROM, bù thân
neck_rom/i18n.py       toàn bộ chữ giao diện (zh mặc định, en)
neck_rom/ui.py         vẽ bằng OpenCV + Pillow (font CJK), màn live / kết quả / lịch sử
neck_rom/app.py        vòng lặp chính, CLI, phím, lưu JSON, log CSV
neck_rom/storage.py    lưu / đọc phiên đo
tools/make_test_video.py  tạo video test từ portrait.jpg
```

## 6. Giới hạn

- **Không phải thiết bị chẩn đoán.** Chỉ để tự theo dõi xu hướng.
- **Cúi / ngửa chưa bù thân** (cần thấy hông, webcam laptop thường không thấy). Đây là future work.
- `trunk_yaw` lấy từ `pose_world_landmarks` (độ sâu ước lượng từ ảnh 2D) nên nhiễu hơn `trunk_roll`.
- Góc đầu dựng từ vài landmark (mắt, trán, cằm), không phải mô hình 3D đầy đủ. Khi quay quá khoảng 60–70°, landmark có thể kém chính xác.
- **Giá trị tham chiếu** (50/60/45/45/80/80° trong `session.py`) là giá trị thường gặp ở người trưởng thành khỏe mạnh; người cao tuổi thường thấp hơn. **Phải trích nguồn cụ thể** (ví dụ AMA Guides, hoặc một bài review chuẩn theo tuổi) và sửa số nếu cần trước khi đưa lên slide.
- Nhóm thử nghiệm nhỏ (2–3 người cao tuổi), chưa đủ để kết luận thống kê.
- **Không commit** dữ liệu `data/` của người cao tuổi lên repo public (`.gitignore` đã loại trừ).

Paper gốc: Chotikkakamthorn et al., *Computers in Biology and Medicine* 195, 110620 (2025). Model của paper dùng để so sánh độ chính xác offline; demo dùng MediaPipe vì chạy realtime trên CPU laptop thường.
