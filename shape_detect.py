import cv2 as cv

# rescaling function
def rescaleFrame(frame , scale):
   width = int(frame.shape[1] * scale)
   height = int(frame.shape[0] * scale)
   dimensions = (width,height)

   return cv.resize(frame , dimensions , interpolation=cv.INTER_AREA)

def getCountours(img, original_f):
    contours, hierarchies = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for a in contours:
        area = cv.contourArea(a)
        area_min = cv.getTrackbarPos('Area','Parameters')
        if area > area_min:
            cv.drawContours(original_f, contours, -1, (0,0,255), 1)


def empty(a):
    pass

cv.namedWindow('Parameters')
cv.resizeWindow('Parameters', 350, 150)
cv.createTrackbar('Thresh 1', 'Parameters', 50, 255, empty)
cv.createTrackbar('Thresh 2', 'Parameters', 100, 255, empty)
cv.createTrackbar('Area', 'Parameters', 6500, 30000, empty)

# Initialize variables for FPS
start_time = cv.getTickCount()
frame_count = 0

vid = cv.VideoCapture('test_data/20221224_204443.mp4')

while True:
    isTrue, frame = vid.read()
    
    # ONLY WHEN READING RECORDED VIDS
    frame = cv.rotate(frame, cv.ROTATE_180)
    # Till this

    frame = rescaleFrame(frame , scale=0.35)
    orignal_f = frame.copy()
    frame = cv.GaussianBlur(frame , (7,7) , 1)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    T1 = cv.getTrackbarPos('Thresh 1','Parameters')
    T2 = cv.getTrackbarPos('Thresh 2','Parameters')
    frame = cv.Canny(frame, T1, T2)
    frame = cv.dilate(frame , (5,5) , iterations=1)
    getCountours(frame,orignal_f)
    
    img = orignal_f
    
    # Calculate elapsed time and fps
    elapsed_time = (cv.getTickCount() - start_time) / cv.getTickFrequency()
    fps = frame_count / elapsed_time
  
    # Display fps on frame
    cv.putText(img, f"FPS: {fps:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

    # DISPLAYS OUTPUT IMAGE
    cv.imshow('Detector',img)

    # if key "f" is pressed then the loop will get terminated 
    if cv.waitKey(20) & 0xFF==ord('f') :
        break
    
    # Increment frame count
    frame_count += 1
