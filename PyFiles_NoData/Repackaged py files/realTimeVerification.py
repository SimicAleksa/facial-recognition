import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from dataForming import preprocess
import testing as test

def verify(model_temp, detection_treshold, verification_threshold):
    # Detection threshold -> prag iznad kojeg se slika smatra identicnom
    # Verification threshold -> proporcija pozitivnih / ukupna kolicina svih pozitivnih uzoraka
    # gledamo folder E:\Desktop\ORI_Projekat\Facial\application_data\verification_images i u njemu se nalazi 45 pozitivnih uzoraka
    results = []
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('application_data', 'verification_images', image))

        # make prediction
        result = model_temp.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

        plt.subplot(1, 2, 1)
        plt.imshow(input_img)
        plt.subplot(1, 2, 2)
        plt.imshow(validation_img)
        plt.text(-10, -10,result, fontsize=22)
        plt.show()

    detection = np.sum(np.array(results) > detection_treshold)
    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
    verified = verification > verification_threshold

    return results, verified


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[40:40 + 250, 200:200 + 250, :]
    cv2.imshow('Verification', frame)  # prikaz prozora live feed-a
    if cv2.waitKey(10) & 0XFF == ord('q'):
        break

    # Verification trigger
    if cv2.waitKey(1) & 0XFF == ord('v'):  # za verifikaciju
        # prvo cuvamo sliku u E:\Desktop\ORI_Projekat\Facial\application_data\input_image
        cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)
        results, verified = verify(test.model, 0.7, 0.7)
        print(verified)
        print(results)
cap.release()  # prekidamo konekciju sa web kamerom
cv2.destroyAllWindows()  # zatvaramo prozor live feed-a

