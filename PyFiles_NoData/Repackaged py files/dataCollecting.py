from dataForming import POS_PATH,ANC_PATH,NEG_PATH
import os
import cv2  # opencv
import uuid  # za kreiranje jedinstvenih imena
# Extractuj lfw.tgz ne u nego direktno odnosno "Extract here" opcija

# Premestanje slika iz lfw-a u negative folder
# samo se jednom pokrece
for directory in os.listdir('lfw'):
    for file in os.listdir(os.path.join('lfw', directory)):
        EX_PATH = os.path.join('lfw', directory, file)
        NEW_PATH = os.path.join(NEG_PATH, file)
        os.replace(EX_PATH, NEW_PATH)


# Velicina slika iz LFW je 250 x 250 pixela, takve nam trebaju i Anchor i Positive
# Prikupljamo slike preko web kamere, za to nam treva opencv


# VideoCapture(<camNUM>) kod camNUM-a moze doci do greske jer gadjamo pogresnu kameru, on se resava promenom vrednosti camNUM-a
cap = cv2.VideoCapture(0)  # dobijamo pristup nasoj web kameri
while cap.isOpened():  # imamo loop frejmova dobijenih od web kamere
    ret, frame = cap.read()  # citamo frejm
    # frejmovi nisu formata 250x250px, zato ih rucno podesavamo
    frame = frame[40:40 + 250, 200:200 + 250, :]

    cv2.imshow('Web cam', frame)  # prikaz prozora live feed-a

    if cv2.waitKey(1) & 0XFF == ord('q'):  # izlazak iz petlje, moramo nekako da obezbedimo izlazak bez Stop 'main'-a
        # waitKey(1) provera na svaku milisekundu
        # 0XFF == ord('q') koju smo tipku pritisnuli
        break

    # za kreiranje positive i anchor slika koristimo lib uuid - koja obezbedjuje jedinstvene nazive
    if cv2.waitKey(1) & 0XFF == ord('a'):  # za pravljenje anchor slika
        # kreiramo jedinstvenu putanju na kojoj cemo da cuvamo sliku
        imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
        # samo cuvanje slike
        cv2.imwrite(imgname, frame)

    if cv2.waitKey(1) & 0XFF == ord('p'):  # za pravljenje posivite slika
        imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, frame)

cap.release()  # prekidamo konekciju sa web kamerom
cv2.destroyAllWindows()  # zatvaramo prozor live feed-a