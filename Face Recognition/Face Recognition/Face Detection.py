import cv2
from managers import WindowManager, CaptureManager

class Cameo(object):
    def __init__(self):
        self._windowManager = WindowManager('Cameo', self.onKeypress)
        self._captureManager = CaptureManager(cv2.VideoCapture(0), self._windowManager, True)
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
    def run(self):
        """Run the main loop."""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame
            # Detect faces
            self.detectFace(frame)
            self._captureManager.exitFrame()
            self._windowManager.processEvents()
            
    def detectFace(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
    def onKeypress(self, keycode):
        """ Handle a keypress.
        space -> Take a screenshot.
        tab -> start/stop recording a screencast.
        escape -> Quit. """
        
        if keycode == 32: # Space
            self._captureManager.writeImage('screenshot.png')
            
        elif keycode == 9: # Tab
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo('screencast.avi')
            else:
                self._captureManager.stopWritingVideo()
                
        elif keycode == 27: # Escape
            self._windowManager.destroyWindow()
            
if __name__ == "__main__":
    Cameo().run()
