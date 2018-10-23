import time

import cv2


def frame_cnter():
    global start_time
    global cnt
    cnt += 1
    # print(start_time)
    if time.time() - start_time >= 0.1:
        print("frame rate %d" % cnt * 10)
        cnt = 0
        start_time = time.time()


class VideoFlow:
    def __init__(self):
        self.video_cap = cv2.VideoCapture(0)
    
    def fetch_frame(self, seq_len=None, show=False):
        """
        获取一个视频序列
        :param seq_len: 序列长度 没有设定则只获取一帧
        :return: 序列
        """
        if seq_len is not None:
            seq = []
            for each in range(seq_len):
                # frame format (W, H, C)
                # crop the black edge
                fm = self.video_cap.read()[1]
                fm = fm[60:fm.shape[0] - 60, :]
                if show:
                    cv2.imshow('show cap', fm)
                cv2.waitKey(1)
                seq.append(fm)
            if show:
                cv2.destroyWindow('show cap')
            return seq
        else:
            fm = self.video_cap.read()[1]
            return fm[60:fm.shape[0] - 60, :]
    
    def release(self):
        self.video_cap.release()


def test():
    video_capture = cv2.VideoCapture(0)
    while True:
        if not video_capture.isOpened():
            print('Unable to load camera.')
            time.sleep(5)
            pass
        
        # Capture frame-by-frame
        start_time = time.time()
        ret, frame = video_capture.read()
        print("internal %f" % (time.time() - start_time))
        # frame = cv2.resize(frame, (255, 255))
        # print(frame.shape)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Draw a rectangle around the faces
        
        # Display the resulting frame
        cv2.imshow('Video', gray)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test()
