from L1DistLayer import L1Dist, tf
from matplotlib import pyplot as plt
import dataForming as df

model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1Dist': L1Dist,
                                                                      'BinaryCrossentropy': tf.losses.BinaryCrossentropy})

test_data = df.test_data
tacnih = 0
broj_slika = 0
for test_input, test_val, y_true in test_data.as_numpy_iterator():
    yhat = model.predict([test_input, test_val])

    for i in range(0, len(test_input)):
        plt.subplot(1, 2, 1)
        plt.imshow(test_input[i])
        plt.subplot(1, 2, 2)
        plt.imshow(test_val[i])
        plt.text(-10, -10, yhat[i], fontsize=22)
        plt.show()
        print(yhat[i], y_true[i])
        if yhat[i] >= 0.8 and y_true[i] == 1.0:
            tacnih = tacnih + 1
        if yhat[i] < 0.8 and y_true[i] == 0.0:
            tacnih = tacnih + 1
        broj_slika = broj_slika + 1

print(tacnih)
print(broj_slika)
tacnost = tacnih / broj_slika
print(tacnost)