from tensorflow.keras.layers import Layer
import tensorflow as tf

class L1Dist(Layer):
    def __init__(self, **kwargs):  # **kwargs nam omogucava da radimo sa ovim slojem kao delom nekog veceg modela
        super().__init__()  # nasledjivanje

    # Efetrivno spajamo Anchor sliku sa ili Positivnom ili Negatrvnom (spajamo <-> odizimamo njihove vrednosti)
    # Funkcija call oznacava akciju odnosno funkciju koja ce se pozvati nad prosledjenim podacima
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
    # ovo je nasoo kao loss funkcija.
