from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Input, Flatten
from tensorflow.keras.models import Model
from L1DistLayer import L1Dist


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
    # velicina je 4096


#
embedding = make_embedding()


# embedding.summary()

def make_siamese_model():
    # Anchor slika/input
    input_image = Input(name='input_img', shape=(100, 100, 3))

    # Validation slika/input
    validation_image = Input(name='validation_img', shape=(100, 100, 3))

    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))

    # Klasifikacioni sloj
    classifier = Dense(1, activation='sigmoid')(distances)  # ovime spajamo 4096 unita u jedan
    # output koji ima vrednost 0 ili 1

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


siamese_model = make_siamese_model()
siamese_model.summary()
