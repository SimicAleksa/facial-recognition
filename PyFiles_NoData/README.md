Facial Verification with a Siamese Network

Dependencies used:
    1.  tensorflow
    2.  tensorflow-gpu
    3.  opencv-python
    4.  matplotlib
    
What are tensors? -> Tensors are the data structure used by machine learning systems, and getting to know them is an essential skill you should build early on. A tensor is a container for numerical data. It is the way we store the information that we'll use within our system.

We are also using https://www.tensorflow.org/guide/keras/functional

Problem resavamo preko Siamese metode sto nam omogucava da koristimo koriscenjem One-shot klasifikaciju.

Open-shot znaci da imamo dva ulazna podatka, odnosno prosledjujemo dve slike u isto vreme koje na kraju prolaze kroz Distance layer.
Distance layer meri slicnost izmedju dve prosledjene slike. Nasu neuronsku mrezu treniramo da odlucuje koliko je velika ta slicnost. Na osnovu dobijenih rezultata kao rezultat dobijamo 1 ako su slike identicne ili 0 ako nisu.

Za negativne slike koristimo LFW http://vis-www.cs.umass.edu/lfw/
Specificno koristimo sledece podatke http://vis-www.cs.umass.edu/lfw/lfw.tgz

Za koriscenje web kamere:
    1. q -> izlazak iz aplikacije
    2. p -> pravljenje positive slike
    3. a -> pravljenje anchor slike
    
Koristimo tensorflow tf.data: Build TensorFlow input pipelines https://www.tensorflow.org/guide/data
Za potrebe koriscenja je neophodno da se napravi minimum 250 anchor i positive slika.
The tf.data API enables you to build complex input pipelines from simple, reusable pieces. 

Naucni rad koriscen kao inspiracija: https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf