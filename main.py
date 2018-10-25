import os
import threading
import time

import cv2

from my_tracker import tracker
from my_tracker import video_test

cache_dir = os.path.join(os.path.dirname(__file__), 'video_cache')
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

store_dir = os.path.join(os.path.dirname(__file__), 'm_store')
if not os.path.exists(store_dir):
    os.makedirs(store_dir)


class Monitor(threading.Thread):
    def __init__(self, video_src, rec=None, save=False):
        threading.Thread.__init__(self)
        self.src = video_src
        self.recs = rec
        self.save = save
        if rec is not None:
            self.recs = list()
            self.recs.append(((0, 0), (160, 160)))
            self.recs.extend(rec)
    
    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter()
        if self.save:
            out.open('output.avi', fourcc, 30.0, (640, 480))
        
        for each_f in range(len(self.src)):
            frame = self.src[each_f]
            if self.recs is not None:
                # print(self.recs[each_f])
                frame = cv2.rectangle(frame, self.recs[each_f][1], self.recs[each_f][0], (255, 0, 0))
            if self.save:
                out.write(frame)
            cv2.imshow('tracking', frame)
            if cv2.waitKey(int((1 / 10) * 1000)) & 0xFF == ord('q'):
                break
        if self.save:
            out.release()


def tracking_process():
    # initialize the camera
    video_flow = video_test.VideoFlow()
    # build up graph
    tracker_o = tracker.Tracker()
    
    print('input the tracking task length: ')
    length = input()
    length = int(length)

    print('input the obj size')
    sz = input()
    sz = int(sz)
    
    # set up start position
    start_time = time.time()
    print('put the tracking obj into the box')
    while time.time() - start_time < 5:
        fm = video_flow.fetch_frame()
        x = fm.shape[0]
        y = fm.shape[1]
        fm = cv2.rectangle(fm,
                           (int(y / 2) - sz / 2, int(x / 2) - sz / 2),
                           (int(y / 2) + sz / 2, int(x / 2) + sz / 2),
                           (255, 0, 0))
        cv2.imshow('tracking', fm)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    
    print('start capturing data')
    input_frames = video_flow.fetch_frame(length, show=True)
    # Monitor(input_frames).run()
    # start tracking, input the frame sequence
    print("start tracking")
    res = tracker_o.tracking(input_frames, bbox_sz=sz)
    #   buffer the result
    #   show the result
    print('show tracking result')
    Monitor(input_frames, res, save=True).start()
    # stop
    video_flow.release()
    cv2.destroyAllWindows()


def save_tflite():
    tracker_o = tracker.Tracker()
    tracker_o.save_tflite()
    


if __name__ == '__main__':
    tracking_process()
    # save_tflite()
    pass

