import os
import urllib.request as request
import numpy as np
import cv2
from PIL import Image
import time

url = 'http://192.168.1.102:8080/shot.jpg'     #url ip_webcam

while True:
    img = request.urlopen(url)
    img_bytes = bytearray(img.read())   #converte immagine in byte
    img_np = np.array(img_bytes, dtype=np.uint8)    #8 bit senza segno (il tipo di dati che scriviamo nell'array numpy)
    #converte l'array in frame (risultato finale)
    frame = cv2.imdecode(img_np, -1)    #-1 flag l'immagine non cambia

    #immage processing riconosce meglio le forme
    frame_cvt = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_blur = cv2.GaussianBlur(frame_cvt, (5, 5), 0)
    frame_edge = cv2.Canny(frame_blur, 30, 50)

    contours, h = cv2.findContours(frame_edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)      #trova i bordi dell'immagine e traccia una linea che li unisce

    if contours:
        max_contour = max(contours, key=cv2.contourArea)        #trova l'area più esterma
        x, y, w, h = cv2.boundingRect(max_contour)      #coordinate del rettangolo
        if cv2.contourArea(max_contour) > 5000:

            #disegna il bordo del rettangolo per delimitare l'oggetto (si può commentare se non si desidera vedere il rettangolo disegnato intorno all'oggetto)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

            object_only = frame[y:y+h, x:x+w]       #taglia l'immagine per scannerizzare solo l'oggetto
            cv2.imshow('Smart Scanner', frame)  #mostra l oggetto acquisito
            key = cv2.waitKey(1)
            if key == ord('s'):
                img_pil = Image.fromarray(object_only)  # for bug fixes use "frame" instead of object_only
                time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
                current_directory = os.path.dirname(os.path.abspath(__file__))
                file_path = os.path.join(current_directory, f'{time_str}.pdf')
                img_pil.save(file_path)
                print(f'Saved {time_str}.pdf')
                break
