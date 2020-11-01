import time, cv2 as cv
import pandas
import datatime
import sys

print(sys.path)

first_frame = None
status_list = [None,None]
times = []
df = pandas.DataFrame(columns=['Start', 'End'])
video = cv.VideoCapture(0)
while True:
    check, frame = video.read()
    status = 0
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = gray
        continue
    delta_frame = cv.absdiff(first_frame, gray)
    thresh_delta = cv.threshold(delta_frame, 30, 255, cv.THRESH_BINARY)[1]
    thresh_delta = cv.dilate(thresh_delta, None, iterations=0)
    (_, cnt, _) = cv.findContours(thresh_delta.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in cnt:
        if cv.contourArea(contour) < 5000:
            continue
        status = 1
        (x, y, w, h) = cv.boundingRect(contour)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
    status_list.append(status)
    status_list = status_list[-2:]
    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())
    cv.imshow('frame', frame)
    cv.imshow('gray', gray)
    cv.imshow('delta frame', delta_frame)
    cv.imshow('thresh_delta', thresh_delta)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
print(status_list)
print(times)
for i in range (0, len(times), 2):
    df = df.append(f"Start:{times[i]},End : {times[i+1]}",ignore_index=True)
df.to_csv("Times.csv")
video.release()
cv.destroyAllWindows()