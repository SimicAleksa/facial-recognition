from testing import tf
import os
import SiaModel as sia
from tensorflow.keras.metrics import Precision, Recall
import dataForming as df

# nasa loss funkcija
binary_cross_loss = tf.losses.BinaryCrossentropy()
#
# # optimajzer
opt = tf.keras.optimizers.Adam(1e-4)  # learning rate je postavljan na 0.0001

siamese_model = sia.siamese_model

# checkpoint callbacks
checkpoint_dir = './training_checkpoints'  # folder u kojem cuvamo nase checkpointove
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

# Proces pri treniranju jednog skupa podataka (jednog batcha) je sledeci:
# 1. Make a prediction
# 2. Calculate loss
# 3. Derive gradients
# 4. Calculate new weights and apply
#
@tf.function
def train_step(batch):  # bazirano na prolasku kroz jedan batch

    with tf.GradientTape() as tape:  # nam ovogucuje "hvatanje" gradijenata iz nase neuronske mreze
        # uzimamo anchor i positive/negative sliku
        X = batch[:2]
        # Get label odnosno vrednost 0 za negativnu ili 1 za pozitivnu
        y = batch[2]

        # Forward pass
        yhat = siamese_model(X, training=True)  # yhat je predikcija odnosno dobijeni rezultat
        loss = binary_cross_loss(y, yhat)  # racunanje lossa

    # Racunanje gradijenata
    grad = tape.gradient(loss, siamese_model.trainable_variables)  # racunaj gradijente po prosledjenoj loss funkciji
    # za celu neuronsku mrezu

    # Racunanje novih tezina i njihovo postavljanje na siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
    # opt optimizer racuna i propagira nove tezinske koeficijente pomocu Adam optimiazcionoig algoritma

    return loss


def train(data_temp, EPOCHS):  # bazirano na prolasku kroz svve batchove
    # epoch -> he beginning of a period in the history of someone or something.
    # loop kroz epochs
    for epoch in range(1, EPOCHS + 1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data_temp))  # inkrementujemo pri prolasku kroz svaki batch
        r = Recall()
        p = Precision()

        # loop kroz sve batchove
        for idx, batch in enumerate(data_temp):
            loss = train_step(batch)
            print(loss)
            yhat_temp = siamese_model.predict(batch[:2])
            r.update_state(batch[2], yhat_temp)
            p.update_state(batch[2], yhat_temp)
            # train_step(batch)
            progbar.update(idx + 1)

        # Save checkpoints
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


# Pocetak treniranja
EPOCHS = 51
train(df.train_data, EPOCHS)
