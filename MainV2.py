import sys
import cv2
import mediapipe as mp
import pyautogui
import math
import time
import threading
from collections import deque

# UI 相關
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QSystemTrayIcon, QMenu, QAction
from PyQt5.QtCore import Qt

# 鍵盤全域偵測 (用來偵測 Ctrl+Q)
import keyboard

# ============ Mediapipe + 手勢滑鼠控制邏輯 ============

class HandMouseController(threading.Thread):
    """
    以執行緒方式跑攝影機 + Mediapipe 偵測手勢。
    在主程式中可隨時設置 self.enabled 來開關功能；若 disabled 則不會移動滑鼠/點擊。
    """

    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self._stop_flag = False  # 用來中止執行緒的旗標
        self.enabled = True      # 是否啟用手勢控制
        self.lock = threading.Lock()

        # Mediapipe 初始化
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        # 取得螢幕解析度
        self.screen_width, self.screen_height = pyautogui.size()

        # 滑鼠移動平滑化參數
        # 在本版本中，將平滑化透過調整 pyautogui.moveTo 的 duration 來實現
        # 不使用滑動平均

        # 狀態
        self.left_button_down = False
        self.right_button_down = False
        self.scroll_y_prev = None
        self.scroll_mode = False

        # 一些閾值 (可透過介面調整)
        self.left_click_threshold = 30
        self.right_click_threshold = 30
        self.scroll_threshold = 5
        self.scroll_sensitivity = 1
        self.scroll_multiplier = 4

        # 手指操作區域距離邊界 100 像素
        self.margin = 100

        # 平滑度相關參數
        self.start_smooth_distance = 200  # 開始平滑化的距離，預設200像素
        self.min_distance = 60            # 拇指與食指距離小於此值時，smoothing=0.05
        self.max_distance = 200           # 拇指與食指距離大於此值時，smoothing=0.01
        self.min_smoothing = 0.01         # 平滑度的最小值
        self.max_smoothing = 0.05         # 平滑度的最大值

    # --- 工具函式 ---

    def calculate_distance(self, point1, point2):
        """計算兩點在歸一化座標 (0~1) 下的歐式距離。"""
        return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

    def is_finger_extended(self, hand_landmarks, finger_tip_id, finger_pip_id):
        """
        以 y 座標判斷某根手指是否伸直 (tip 的 y 小於 pip 表示手指在上方)。
        依實際情況可能要再調整判斷方式。
        """
        return hand_landmarks.landmark[finger_tip_id].y < hand_landmarks.landmark[finger_pip_id].y

    def all_fingers_extended(self, hand_landmarks):
        """
        簡易判斷：食指(8/6)、中指(12/10)、無名指(16/14)、小拇指(20/18) 是否都伸直。
        拇指不納入判斷，也可自行修改。
        """
        fingers = [
            self.is_finger_extended(hand_landmarks, 8, 6),   # 食指
            self.is_finger_extended(hand_landmarks, 12, 10), # 中指
            self.is_finger_extended(hand_landmarks, 16, 14), # 無名指
            self.is_finger_extended(hand_landmarks, 20, 18), # 小拇指
        ]
        return all(fingers)

    def stop(self):
        self._stop_flag = True

    def run(self):
        """
        執行緒主函式：讀取攝影機影像 + hand tracking + 滑鼠控制
        """
        cap = cv2.VideoCapture(0)

        # 禁用 PyAutoGUI 的 Fail-Safe 功能
        pyautogui.FAILSAFE = False  # 注意：禁用 Fail-Safe 可能導致無法通過滑鼠移動到螢幕角落來停止腳本

        while not self._stop_flag:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = frame.shape

            # ROI 參數
            x_min, y_min = self.margin, self.margin
            x_max, y_max = frame_width - self.margin, frame_height - self.margin
            roi_w = x_max - x_min
            roi_h = y_max - y_min

            # Mediapipe 偵測
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb_frame)

            # 初始化 smoothing 為預設值
            smoothing = 0.01  # 預設平滑度

            if result.multi_hand_landmarks:
                # 只看第一隻手
                hand_landmarks = result.multi_hand_landmarks[0]
                # 繪製關鍵點 (for debug)
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # 關節
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
                middle_tip = hand_landmarks.landmark[12]

                # 判斷食指/中指是否伸直、併攏
                index_extended = self.is_finger_extended(hand_landmarks, 8, 6)
                middle_extended = self.is_finger_extended(hand_landmarks, 12, 10)
                close_dist = self.calculate_distance(index_tip, middle_tip) * frame_width
                fingers_close = close_dist < 30

                # 捲動模式判斷
                if index_extended and middle_extended and fingers_close:
                    self.scroll_mode = True
                    current_scroll_y = int(index_tip.y * self.screen_height)
                    if self.scroll_y_prev is not None:
                        delta_y = current_scroll_y - self.scroll_y_prev
                        if abs(delta_y) > self.scroll_threshold and self.enabled:
                            scroll_amount = int(delta_y / self.scroll_sensitivity) * self.scroll_multiplier
                            pyautogui.scroll(-scroll_amount)
                    self.scroll_y_prev = current_scroll_y
                else:
                    self.scroll_mode = False
                    self.scroll_y_prev = None

                # 動態計算平滑度 smoothing
                # -------------------------------------------------------------------
                thumb_index_dist = self.calculate_distance(thumb_tip, index_tip) * frame_width

                # 計算平滑度：
                # 當拇指與食指距離 > start_smooth_distance 時，smoothing = 0.01
                # 當距離 < min_distance 時，smoothing = 0.05
                # 當距離在 min_distance 和 start_smooth_distance 之間，線性插值
                if thumb_index_dist > self.start_smooth_distance:
                    smoothing = 0.01
                elif thumb_index_dist < self.min_distance:
                    smoothing = self.max_smoothing
                else:
                    # 線性插值
                    ratio = (self.start_smooth_distance - thumb_index_dist) / (self.start_smooth_distance - self.min_distance)
                    smoothing = self.min_smoothing + ratio * (self.max_smoothing - self.min_smoothing)
                    smoothing = round(smoothing, 3)  # 保留三位小數

                # 確保 smoothing 在合理範圍內
                smoothing = max(self.min_smoothing, min(self.max_smoothing, smoothing))
                # -------------------------------------------------------------------

                # 不在捲動模式才控制滑鼠移動
                if not self.scroll_mode:
                    # ROI 映射 -> 螢幕
                    px = index_tip.x * frame_width
                    py = index_tip.y * frame_height
                    # clamp 到 ROI
                    px = max(x_min, min(x_max, px))
                    py = max(y_min, min(y_max, py))
                    # 線性映射到整個螢幕
                    mapped_x = int((px - x_min) / roi_w * self.screen_width)
                    mapped_y = int((py - y_min) / roi_h * self.screen_height)

                    # 根據平滑度調整滑鼠移動的 duration
                    duration = smoothing  # duration 設為 0.01 至 0.05 秒之間

                    # 移動滑鼠
                    if self.enabled:
                        try:
                            pyautogui.moveTo(mapped_x, mapped_y, duration=duration)
                        except pyautogui.FailSafeException:
                            # 如果仍然觸發 Fail-Safe，可以選擇忽略或進行其他處理
                            print("PyAutoGUI FailSafeException ignored.")
                        except Exception as e:
                            print(f"Unexpected error during moveTo: {e}")

                # 判斷左鍵 (大拇指 與 食指)
                thumb_index_dist_click = self.calculate_distance(thumb_tip, index_tip) * frame_width
                if thumb_index_dist_click < self.left_click_threshold:
                    if not self.left_button_down and self.enabled:
                        pyautogui.mouseDown(button='left')
                        self.left_button_down = True
                else:
                    if self.left_button_down and self.enabled:
                        pyautogui.mouseUp(button='left')
                        self.left_button_down = False

                # 判斷右鍵 (大拇指、食指、中指都靠近)
                thumb_middle_dist = self.calculate_distance(thumb_tip, middle_tip) * frame_width
                if (thumb_index_dist_click < self.right_click_threshold and
                    thumb_middle_dist < self.right_click_threshold and
                    close_dist < self.right_click_threshold):
                    if not self.right_button_down and self.enabled:
                        pyautogui.mouseDown(button='right')
                        self.right_button_down = True
                else:
                    if self.right_button_down and self.enabled:
                        pyautogui.mouseUp(button='right')
                        self.right_button_down = False

            # 只顯示 ROI 方框以做 debug
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # 在影像上顯示目前是否 Enabled，以及當前 smoothing
            status_text = "Enabled" if self.enabled else "Disabled"
            cv2.putText(frame, f"Gesture Control: {status_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Smoothing (duration): {smoothing}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            cv2.imshow("Hand Mouse Controller", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# ============ PyQt5 介面 + 系統托盤程式 ============

class MainWindow(QtWidgets.QMainWindow):
    """
    主視窗：簡單放幾個可調整的設定 (滑鼠靈敏度、捲動靈敏度...)，並可最小化到系統托盤
    """
    def __init__(self, controller: HandMouseController):
        super().__init__()
        self.controller = controller
        self.initUI()

    def initUI(self):
        self.setWindowTitle("手勢滑鼠控制 - 設定視窗")
        self.setFixedSize(500, 500)  # 擴大視窗大小至500x500

        # 整個視窗放一個中央Widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout()
        central_widget.setLayout(layout)

        # 左鍵距離閾值
        self.left_threshold_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.left_threshold_slider.setRange(10, 100)
        self.left_threshold_slider.setValue(self.controller.left_click_threshold)
        self.left_threshold_slider.valueChanged.connect(self.onLeftThresholdChanged)
        layout.addWidget(QtWidgets.QLabel("左鍵點擊距離閾值"))
        layout.addWidget(self.left_threshold_slider)

        # 右鍵距離閾值
        self.right_threshold_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.right_threshold_slider.setRange(10, 100)
        self.right_threshold_slider.setValue(self.controller.right_click_threshold)
        self.right_threshold_slider.valueChanged.connect(self.onRightThresholdChanged)
        layout.addWidget(QtWidgets.QLabel("右鍵點擊距離閾值"))
        layout.addWidget(self.right_threshold_slider)

        # 滾動靈敏度
        self.scroll_sens_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.scroll_sens_slider.setRange(1, 20)
        self.scroll_sens_slider.setValue(self.controller.scroll_sensitivity)
        self.scroll_sens_slider.valueChanged.connect(self.onScrollSensitivityChanged)
        layout.addWidget(QtWidgets.QLabel("滾動靈敏度（數值越小越敏感）"))
        layout.addWidget(self.scroll_sens_slider)

        # 最大平滑度
        self.max_smoothing_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.max_smoothing_slider.setRange(1, 100)  # 增加範圍以允許更細緻的調整
        self.max_smoothing_slider.setValue(int(self.controller.max_smoothing * 100))  # 轉換為整數
        self.max_smoothing_slider.valueChanged.connect(self.onMaxSmoothingChanged)
        layout.addWidget(QtWidgets.QLabel("最大平滑度 (0.01 - 0.05)"))
        layout.addWidget(self.max_smoothing_slider)

        # 最小手指距離
        self.min_dist_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.min_dist_slider.setRange(10, 100)
        self.min_dist_slider.setValue(self.controller.min_distance)
        self.min_dist_slider.valueChanged.connect(self.onMinDistanceChanged)
        layout.addWidget(QtWidgets.QLabel("最小手指距離（像素，越小越敏感）"))
        layout.addWidget(self.min_dist_slider)

        # 開始平滑化的距離
        self.start_smooth_distance_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.start_smooth_distance_slider.setRange(100, 300)
        self.start_smooth_distance_slider.setValue(self.controller.start_smooth_distance)
        self.start_smooth_distance_slider.valueChanged.connect(self.onStartSmoothDistanceChanged)
        layout.addWidget(QtWidgets.QLabel("開始平滑化的距離（像素）"))
        layout.addWidget(self.start_smooth_distance_slider)

        # 調整後的滑動平滑度參數
        layout.addWidget(QtWidgets.QLabel("平滑度會根據手指距離動態調整。"))

        # 關於提示
        layout.addWidget(QtWidgets.QLabel("按 Ctrl+Q 可啟用/停用手勢控制"))
        layout.addStretch()

    # ---- Slider 事件 ----
    def onLeftThresholdChanged(self, val):
        self.controller.left_click_threshold = val

    def onRightThresholdChanged(self, val):
        self.controller.right_click_threshold = val

    def onScrollSensitivityChanged(self, val):
        self.controller.scroll_sensitivity = val

    def onMaxSmoothingChanged(self, val):
        # 將滑桿值從 1-100 轉換為 0.01 - 0.05
        self.controller.max_smoothing = val / 100.0

    def onMinDistanceChanged(self, val):
        self.controller.min_distance = val

    def onStartSmoothDistanceChanged(self, val):
        """
        調整開始平滑化的距離（start_smooth_distance）
        """
        self.controller.start_smooth_distance = val

    # ---- 覆寫 closeEvent 以防止直接關掉 ----
    def closeEvent(self, event):
        """
        點右上角 X 時，不是真正退出，而是隱藏到系統托盤。
        """
        event.ignore()
        self.hide()

class SystemTrayApp(QtWidgets.QApplication):
    """
    建立系統托盤圖示，並控制主視窗的顯示/隱藏
    """
    def __init__(self, argv, controller: HandMouseController):
        super().__init__(argv)
        self.controller = controller

        # 建立主視窗
        self.main_window = MainWindow(controller)

        # 建立系統托盤
        self.tray_icon = QSystemTrayIcon(self)
        icon_path = "icon.png"  # 請將此路徑替換為您的圖示檔案路徑
        if QtCore.QFile.exists(icon_path):
            icon = QtGui.QIcon(icon_path)
        else:
            icon = QtGui.QIcon()  # 使用預設圖示
            print(f"警告: 圖示檔案 '{icon_path}' 未找到，將使用預設圖示。")
        self.tray_icon.setIcon(icon)
        self.tray_icon.setToolTip("手勢滑鼠控制器")
        
        # 在托盤上定義選單
        tray_menu = QMenu()
        show_action = QAction("顯示 / 隱藏 設定視窗", self)
        show_action.triggered.connect(self.toggleMainWindow)
        tray_menu.addAction(show_action)

        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.exitApp)
        tray_menu.addAction(exit_action)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

    def toggleMainWindow(self):
        """
        顯示或隱藏主視窗
        """
        if self.main_window.isVisible():
            self.main_window.hide()
        else:
            self.main_window.show()
            self.main_window.activateWindow()

    def exitApp(self):
        """
        結束應用程式
        """
        self.controller.stop()
        self.quit()

def toggleEnabled(controller: HandMouseController):
    """
    全域按鍵 Ctrl+Q 呼叫此函式；切換手勢控制功能的啟動與否
    """
    with controller.lock:
        controller.enabled = not controller.enabled
        print(f"[Ctrl+Q] Gesture Control Enabled = {controller.enabled}")

def main():
    # 建立手勢控制執行緒
    controller = HandMouseController()
    controller.start()

    # 監聽 Ctrl+Q 全域按鍵
    keyboard.add_hotkey("ctrl+q", lambda: toggleEnabled(controller))

    # 啟動 PyQt5
    app = SystemTrayApp(sys.argv, controller)
    app.main_window.show()  # 一開始顯示設定視窗，如果想直接隱藏可註解掉
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
