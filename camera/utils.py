# Probe the cameras, return indeces, fourcc, default resolution
import cv2 
def probeCameras(numcams: int = 10):
    # check for up to 10 cameras
    index = 0
    arr = []
    i = numcams
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            tmp = cap.get(cv2.CAP_PROP_FOURCC)
            fourcc = "".join([chr((int(tmp) >> 8 * i) & 0xFF) for i in range(4)])
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap.release()
            arr.append({"index": index, "fourcc": fourcc, "width": width, "height": height})
        index += 1
        i -= 1
    return arr
