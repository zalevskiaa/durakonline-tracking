import sys
import cv2
from PyQt5.QtWidgets import \
    QApplication, QLabel, QHBoxLayout, \
    QWidget, QPushButton
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, pyqtSignal

import time
from PyQt5.QtCore import QThread

from PyQt5.QtCore import QMutex, QMutexLocker

from durak_detector import CardsDetector
from game_tracker import GameTracker
from settings import CAMERA_SOURCE


class Atomic:
    def __init__(self):
        self.mutex = QMutex()
        self.object = None

    def set(self, obj):
        with QMutexLocker(self.mutex):
            self.object = obj

    def get(self):
        with QMutexLocker(self.mutex):
            result = self.object
            self.object = None
            return result


class FramesProcessor:
    def __init__(self):
        self.cards_detector = CardsDetector()
        self.game_state = GameTracker()

    def process(self, frames: list) -> list:
        if frames:
            preds = self.cards_detector.predict(frames)
            batch_boxes = preds['batch_boxes']
            batch_ranks = preds['batch_ranks']
            batch_suits = preds['batch_suits']

            batch_players_boxes = preds['batch_players_boxes']
            batch_players_take = preds['batch_players_take']

            for i, frame in enumerate(frames):
                boxes = batch_boxes[i]
                ranks = batch_ranks[i]
                suits = batch_suits[i]

                players_boxes = batch_players_boxes[i]
                players_take = batch_players_take[i]

                # line devides table and player's cards

                h, w, _ = frame.shape
                y_table_bottom = round(900 * h / 1280)
                frame = cv2.line(
                    frame,
                    (0, y_table_bottom),
                    (w, y_table_bottom),
                    (0, 255, 0),
                    2
                )

                # detected players and classified "I take" or not

                for box, is_take in zip(players_boxes, players_take):
                    x1, y1, x2, y2 = box
                    frame = cv2.rectangle(
                        frame, (x1, y1), (x2, y2),
                        color=(0, 255, 0),
                        thickness=2
                    )
                    text = {'0': 'default', '1': 'take'}[is_take]
                    frame = cv2.putText(
                        frame,
                        text=text,
                        org=(x1, y1),
                        fontFace=0,
                        fontScale=0.7,
                        color=(0, 200, 0),
                        thickness=2
                    )

                # # me box
                # mbs2 = 0.07 * w
                # me_box = (
                #     round(0.5 * w - mbs2), round(0.885 * h - mbs2),
                #     round(0.5 * w + mbs2), round(0.885 * h + mbs2),
                # )

                # p1_box = (
                #     round((0.5 - 0.15) * w - mbs2), round(0.15 * h - mbs2),
                #     round((0.5 - 0.15) * w + mbs2), round(0.15 * h + mbs2),
                # )
                # p2_box = (
                #     w - p1_box[2], p1_box[1],
                #     w - p1_box[0], p1_box[3],
                # )
                # for box in [me_box, p1_box, p2_box]:
                #     x1, y1, x2, y2 = box
                #     frame = cv2.rectangle(
                #         frame, (x1, y1), (x2, y2),
                #         color=(0, 255, 0),
                #         thickness=2
                #     )

                # detected cards and classified ranks and suits

                for box, r, s in zip(boxes, ranks, suits):
                    x1, y1, x2, y2 = box
                    frame = cv2.rectangle(
                        frame, (x1, y1), (x2, y2),
                        color=(0, 255, 0),
                        thickness=2
                    )
                    frame = cv2.putText(
                        frame,
                        text=f'{r}{s}',
                        org=(x1, y1),
                        fontFace=0,
                        fontScale=0.7,
                        color=(0, 200, 0),
                        thickness=2
                    )

                # send update to game_state

                hardcode_geom = {
                    'y_table_bottom': y_table_bottom,
                }

                self.game_state.update(boxes, ranks, suits,
                                       players_boxes, players_take,
                                       hardcode_geom=hardcode_geom)
                state = self.game_state.get_summary()
                frames[i] = frame, state

        return frames


class WorkerThread(QThread):
    frame_done = pyqtSignal()

    def __init__(self, frame_in, frame_out):
        super().__init__()
        self.running = True

        self.frame_in: Atomic = frame_in
        self.frame_out: Atomic = frame_out

        self.frames_processor = FramesProcessor()

    def run(self):
        while self.running:
            frame = self.frame_in.get()

            if frame is not None:
                frames = [frame]
                frames = self.frames_processor.process(frames)

                frame = frames[0]
                self.frame_out.set(frame)

                self.frame_done.emit()

            time.sleep(0.1)

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

    def on_newgame_clicked(self):
        print('worker: new game clicked')
        self.frames_processor.game_state.start_new_game()


class App(QWidget):
    def __init__(self, video_source=0):
        super().__init__()

        self.cap = cv2.VideoCapture(video_source)
        self.frame_in: Atomic = Atomic()
        self.frame_out: Atomic = Atomic()

        self.frame_label = QLabel(self)
        self.text_label = QLabel(self)
        self.text_label.setText('some text')
        self.text_label.setFont(QFont('MS Shell Dlg 2', 16))

        self.button_new_game = QPushButton('new game', self)

        layout = QHBoxLayout()
        layout.addWidget(self.frame_label)
        layout.addWidget(self.text_label)
        layout.addWidget(self.button_new_game)
        self.setLayout(layout)

        self.cap_fps = 60
        self.fps = 7

        self.timer = QTimer()
        self.timer.timeout.connect(self.capture_frame)
        self.timer.start(1000 // self.cap_fps)
        self.frame_cap = None

        self.timer_send = QTimer()
        self.timer_send.timeout.connect(self.send_frame)
        self.timer_send.start(1000 // self.fps)

        # self.timer_view = QTimer()
        # self.timer_view.timeout.connect(self.view_frame)
        # self.timer_view.start(1000 // self.fps)

        self.worker = WorkerThread(self.frame_in, self.frame_out)
        self.worker.frame_done.connect(self.view_frame)
        self.worker.start()

        self.button_new_game.clicked.connect(self.worker.on_newgame_clicked)

    def capture_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (720, 1280))
            self.frame_cap = frame.copy()

    def send_frame(self):
        if self.frame_cap is not None:
            self.frame_in.set(self.frame_cap)
            self.frame_cap = None

    def view_frame(self):
        frame_state = self.frame_out.get()
        if frame_state is not None:
            frame, state = frame_state
            self.set_frame(frame)
            self.text_label.setText(state)

    def set_frame(self, frame):
        scale = 900 / frame.shape[0]
        dsize = round(scale * frame.shape[1]), round(scale * frame.shape[0])
        frame = cv2.resize(frame, dsize)

        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.frame_label.setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = App(CAMERA_SOURCE)
    window.show()
    sys.exit(app.exec_())
