#######################################################################
############################ Importovi# ###############################
#######################################################################

# Standardni importovi
import cv2  # opencv
import os  # lib za operativni sistem potrapno pri kreiranju strukture foldera
import random
import numpy as np
from matplotlib import pyplot as plt
# imamo opciju plt.imshow() nam daje mogucnost da prikazujemo podatke kao sliku
import uuid  # za kreiranje jedinstvenih imena

# TensorFlow importovi
# koristimo tensorflow fuctional api
from tensorflow.keras.models import Model  # nama verovatno najbitniji
# Kad definisemo model preko functional api-ja imamo Model(inputs= <ulazni podaci>,outpputs= <izlazni podaci>)
# Gradimo nasu neuronsku mrezu i efektivno postavljano nase inputove i outputove
# primer -> Model(inputs=[inputImage, verificationImage], outputs=[1,0])  output nam je sloj 0,1

from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPool2D, Input, \
    Flatten  # za neuronsku morezu su potrebni slojevi, ovim importovima dovijamo razne vrste slojeva
# Layer nam omogucava da definisemo custom sloj
# primer -> class L1Dist(Layer)
# Conv2D nam omogucuje da radimo konvolucije
# Dense nam omogucuje da imamo FULLY CONNECTED LAYER (svaki cvor trenutnog sloja je povezan sa svakim cvorom prethodnog sloja)
# MaxPoolling2D nam omogucuje da spojimo nase slojeve cime dobijemo smanjenje dobijene kolicine podataka, donosno smanjujemo kolicinu prosledjenih podataka sledecem sloju (uziva max vrednosti preko nekog predodredjenog praga)
# Input pomocu njega definisemo sta prosledjujemo nasem modelu
# Flatten uzima podatke dobijene iz prethodnog sloja i "spljoska" ih u jednu dimenziju zarad prosledjivanja CNN podataka DENSE layeru
import tensorflow as tf

#######################################################################
############################ GPU Growth ###############################
#######################################################################
# Zarad izbvegavanja errora za nedostatak memorije postavljamo ogranicenja na grafickoj
########
# gpus = tf.config.experimental.list_physical_devices('GPU')
# # print(gpus)
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
########
# problem je sledeci ne radi na integrisanim grafickim kartama !!! ali boze moj ne moze da skodi ako ostane


#######################################################################
############################ Kreiranje foldera ########################
#######################################################################

# kreiramo tri foldera: 1 - Anchor    2 - Positive    3 - Negative
# Pri verifikacijji lica prosledjujemo dve slike jedna Ancor i jedna Pozitivna ili Negativna
# Anchor je na primer slika koju dobijamo preko kamere, ona koju zelimo da verifikujemo
# Positive je slika koja je identicna nasoj verifikacionoj dok je Negative slika koja nije identicna nasoj verifikacionoj
# U nasem slucaju sve Negativne slike dobijamo iz repozitorijuma LFW (Labeled Faces in the Wild)

# Za kreiranje foldera koristimo nas os import

# Podesavanje putanja
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

######## Kreiranje foldera ########## samo se jednom pokrece
# os.makedirs(POS_PATH)
# os.makedirs(NEG_PATH)
# os.makedirs(ANC_PATH)


#######################################################################
############################ LFW  #####################################
#######################################################################

# Extractuj lfw.tgz ne u nego direktno odnosno "Extract here" opcija

# Premestanje slika iz lfw-a u negative folder
# samo se jednom pokrece
# for directory in os.listdir('lfw'):
#     for file in os.listdir(os.path.join('lfw', directory)):
#         EX_PATH = os.path.join('lfw', directory, file)
#         NEW_PATH = os.path.join(NEG_PATH, file)
#         os.replace(EX_PATH, NEW_PATH)


#######################################################################
################## Positive and Anchor Collecting  ####################
#######################################################################

# Velicina slika iz LFW je 250 x 250 pixela, takve nam trebaju i Anchor i Positive
# Prikupljamo slike preko web kamere, za to nam treva opencv


# VideoCapture(<camNUM>) kod camNUM-a moze doci do greske jer gadjamo pogresnu kameru, on se resava promenom vrednosti camNUM-a
# cap = cv2.VideoCapture(0)  # dobijamo pristup nasoj web kameri
# while cap.isOpened():  # imamo loop frejmova dobijenih od web kamere
#     ret, frame = cap.read()  # citamo frejm
#     # frejmovi nisu formata 250x250px, zato ih rucno podesavamo
#     frame = frame[40:40 + 250, 200:200 + 250, :]
#
#     cv2.imshow('Web cam', frame)  # prikaz prozora live feed-a
#
#     if cv2.waitKey(1) & 0XFF == ord('q'):  # izlazak iz petlje, moramo nekako da obezbedimo izlazak bez Stop 'main'-a
#         # waitKey(1) provera na svaku milisekundu
#         # 0XFF == ord('q') koju smo tipku pritisnuli
#         break
#
#     # za kreiranje positive i anchor slika koristimo lib uuid - koja obezbedjuje jedinstvene nazive
#     if cv2.waitKey(1) & 0XFF == ord('a'):  # za pravljenje anchor slika
#         # kreiramo jedinstvenu putanju na kojoj cemo da cuvamo sliku
#         imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
#         # samo cuvanje slike
#         cv2.imwrite(imgname, frame)
#
#     if cv2.waitKey(1) & 0XFF == ord('p'):  # za pravljenje posivite slika
#         imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
#         cv2.imwrite(imgname, frame)
#
# cap.release()  # prekidamo konekciju sa web kamerom
# cv2.destroyAllWindows()  # zatvaramo prozor live feed-a

#######################################################################
##################### Procesiranje slika  #############################
#######################################################################

# tf.data.Dataset.list_files(<>) -> pribavljamo sve slike iz foldera
# kreiramo pipelineove
anchor = tf.data.Dataset.list_files(ANC_PATH + '\*.jpg').take(250)
positive = tf.data.Dataset.list_files(POS_PATH + '\*.jpg').take(250)
negative = tf.data.Dataset.list_files(NEG_PATH + '\*.jpg').take(250)


# I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
# To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
# Nije error nego mogucnost poboljsanja rada tensorflow-a


# funkcija preprocess nam vraca numpy vrednost prosledjene slike
def preprocess(file_path):
    # citanje slike sa putanje file_path
    byte_img = tf.io.read_file(file_path)

    # ucitavanje/dekodiranje dobijene slike
    img = tf.io.decode_jpeg(byte_img)

    # resizujemo nasu sliku na format 100x100px
    # MOZES DA PROPRATIS NAUCNI RAD NAPISAN U README.md -> vrednosti koje su tamo zadate su 105x105px
    img = tf.image.resize(img, (100, 100))

    # skaliranje slike
    # vrednost pixela moze biti od 0 do 255 mi efektivno deljenjem nase slike sa 255 svodimo vrednost izmedju 0 i 1 sto je nama potrebno
    img = img / 255.0
    return img


#######################################################################
########### Kreiranje setova slika / nase baze podataka  ##############
#######################################################################

## (anchor,positive) => 1,1,1,1,1
## (anchor,negative) => 0,0,0,0,0

# .zip nam daje mogucnost da iteriramo kroz sve tri liste(ancgor,positive,tf.data.Data.....)
# od nasih lista smo formirali tuple
# za positivne slike smo stavili tf.ones zato sto postatramo da nam je identicna slika 1 a ona koja nije identicna 0
# u sustini mi oznacavamo da se radi o identicnim odnosno neidenticnim slikama
positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
# na kraju spajamo positives i negatives setove podataka u jednu veliku bazu podataka
data = positives.concatenate(negatives)


# koristimo za prosledjivanje podataka preprocess funkciji, odnosno podatke iz 'data' seta podataka raspakujemo
def preprocess_twin(input_img, validation_img, label):
    return preprocess(input_img), preprocess(validation_img), label


# pri prosledjivanju podataka koristiti preprocess_twin(*podatak) *-> nam raspakuje tuple


# Building Dataloader pipeline   uzimamo podatke ........??????????
data = data.map(preprocess_twin)  # parametar za .map je funkcija koja mapira podatke ?!?!?!?!?!? mozda
data = data.cache()
data = data.shuffle(buffer_size=1024)  # mesamo slike da bi izbegli 250 redom pozitivnih i negativnih +
# zarad treniranja i testiranja jer delimo bazu na dva dela

# podela baze na particije za treniranje i testiranje
train_data = data.take(round(len(data) * 0.7))  # uzimamo 70% naseg napravljenog seta podataka za treniranje
train_data = train_data.batch(16)  # podatke pakujemo u serijama/skupovima bečovima od 16 komada
train_data = train_data.prefetch(8)  # pocinje pretprocesiranje sledeceg seta sliga da ne bi doveli\
# nasu neuronsku mrezu do bottleneck-ovanja

# .take uzima prvih toliko i toliko, zato moramo da preskocimo prvih 70% za pravljenje testne particije
test_data = data.skip(round(len(data) * 0.7))
test_data = test_data.take(round(len(data) * 0.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)


#######################################################################
################### Embedding i Distance layeri  ######################
#######################################################################

# embedding sloj nam pretvara nasu sliku u neke RAW podatke zarad prosledjivanja podataka neuronskoj mrezi
# koja tek onda moze da utvrdi da li se radi o verifikovanoj osobi ili ne odnosno neuronska mreza ne zna
# da barata sa slikama pa mi te slike pretvaramo u podatke njoj razumljive

# L1 Distance layer koristimo za samo uporedjivanje slika odnosno embedingova u nasoj neuronskoj mrezi

def make_embedding():
    # u sustini prolazimo kroz naucni rad i kreiramo slojeve kao sto je tamo prikazana na slici odnosno na Figure 4
    inp = Input(shape=(100, 100, 3), name='input_image')  # u naucnom radu je stavljano 105x105

    # blok predstavlja jednu celinu nase mreze i on se sastoji iz konvolucije + relu-1 + maxpooli-inga
    # blokovi mogu biti razlicitih oblika, i oni se ponavljaju (msm na blokove ne na njihove oblike)

    # Prvi blok
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)  # (inp) ide na kraj jer ovako vezujemo nase slojeve
    m1 = MaxPool2D(64, (2, 2), padding='same')(c1)  # MaxPool2D u ovom slucaju uzima oblast 2x2 i iz nje bira
    # najvecu vrednost, samim tim vrednost tog 2x2 bloka postaje ta vrednost

    # Drugi blok
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)  # (m1) ide na kraj jer ga prosledjujemo c2-jci
    m2 = MaxPool2D(64, (2, 2), padding='same')(c2)

    # Treci blok
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)  # (m1) ide na kraj jer ga prosledjujemo c2-jci
    m3 = MaxPool2D(64, (2, 2), padding='same')(c3)

    # Cetvrti/finalni blok
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)  # Flatten je potpuna konekcima medju slojevima
    # ili Flatten uzima nas 3d prostor i spljoska ga u 1d prostor npr prostor c3-a je 128x4x4 on ga spljoska u jedan
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=[inp], outputs=[d1], name='embedding')  # ovim nase slike pretvaramo u VEKTORE


model = make_embedding()
model.summary()

