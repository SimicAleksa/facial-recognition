import os
from L1DistLayer import tf
# Podesavanje putanja
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')
#
# ####### Kreiranje foldera ########## samo se jednom pokrece
os.makedirs(POS_PATH)
os.makedirs(NEG_PATH)
os.makedirs(ANC_PATH)

# kreiramo pipelineove
anchor = tf.data.Dataset.list_files(ANC_PATH + '\*.jpg').take(100)  # povacaj kasnije ako bude trebalo
positive = tf.data.Dataset.list_files(POS_PATH + '\*.jpg').take(100)
negative = tf.data.Dataset.list_files(NEG_PATH + '\*.jpg').take(100)

# funkcija preprocess nam vraca numpy vrednost prosledjene slike
# mi citamo file sa putanje zatim taj kodirani podatak dekodujemo u jpeg => resajzujemo taj jpeg na nama potrebne dimenzije i delimo dobijenu sliku sa 255 da bi sveli njenu vrednost na vrednsot izmedju 0 i 1
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



