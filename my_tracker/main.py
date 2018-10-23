import os
import threading
import time

import cv2

from my_tracker import tracker
from my_tracker import video_test

cache_dir = os.path.join(os.path.dirname(__file__), 'video_cache')
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)


class Monitor(threading.Thread):
    def __init__(self, video_src, rec=None):
        threading.Thread.__init__(self)
        self.src = video_src
        self.recs = rec
        if rec is not None:
            self.recs = list()
            self.recs.append(((0, 0), (160, 160)))
            self.recs.extend(rec)
    
    def run(self):
        for each_f in range(len(self.src)):
            frame = self.src[each_f]
            if self.recs is not None:
                print(self.recs[each_f])
                frame = cv2.rectangle(frame, self.recs[each_f][0], self.recs[each_f][1], (255, 0, 0))
                cv2.imwrite(os.path.join(cache_dir, '%d.jpg' % each_f), frame)
            cv2.imshow('tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    
    # initialize the camera
    video_flow = video_test.VideoFlow()
    # build up graph
    tracker = tracker.Tracker()
    # set up start position
    start_time = time.time()
    while time.time() - start_time < 2:
        fm = video_flow.fetch_frame()
        # fm = cv2.resize(fm,(400, 300))
        fm = cv2.rectangle(fm, (0, 0), (160, 160), (255, 0, 0))
        cv2.imshow('tracking', fm)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    print('start tracking')
    input_frames = video_flow.fetch_frame(80, show=True)
    Monitor(input_frames).start()
    # start tracking, input the frame sequence
    res = tracker.tracking(input_frames)
    #   buffer the result
    #   show the result
    Monitor(input_frames, res).start()
    # stop
    video_flow.release()
    cv2.destroyAllWindows()
    pass
