from django.urls import reverse
import pandas as pd
from sklearn.pipeline import Pipeline
from tensorflow.python.keras.models import load_model
from keras import backend as K
from appPractica4SvmCnn.Logica import modelos
import pickle
import keras
from django.shortcuts import render
from PIL import Image
import numpy as np

class modelos():
    """Clase modelo Preprocesamiento y SNN"""


''' 
    def subirImagenPreprocesadaSVM(request):
        if request.method == 'POST':
            form = ImagenForm(request.POST, request.FILES)
            if form.is_valid():
                # Guarda la imagen
                imagen = form.cleaned_data['imagen']
                ruta = 'Pr4_SVM_CNN/Recursos/Imagenes/' + imagen.name
                with open(ruta, 'wb+') as destination:
                    for chunk in imagen.chunks():
                        destination.write(chunk)

                # Procesa la imagen
                imagenOr = Image.open(ruta)
                imagenPIL = imagenOr.resize((32, 32))
                imagen_np = np.array(imagenPIL)

                return render(request, 'resultado.html', {'resultado': tu_resultado})
        else:
            form = ImagenForm()

        return render(request, 'formulario.html', {'form': form})
'''


def cargarNN(nombreArchivo):
    model = keras.models.load_model(nombreArchivo + '.h5')
    print("Red Neuronal Cargada desde Archivo")
    return model
def cargarObjeto(nombreArchivo):
        with open(nombreArchivo + '.pickle', 'rb') as handle:
            pipeline  = pickle.load(handle)
            print("Objeto Cargado desde Archivo")
        return pipeline

def prediccionSVM(imagen):
        #ImagenAplanada
        imagen_flatten = imagen.reshape(-1)
        imagen_flatten = imagen_flatten / 255
        SVM_Predict=cargarObjeto("Recursos/SVMWeb")
        prediccion_SVM = SVM_Predict.predict(imagen_flatten.reshape(1, -1))
        print("Predicción SVM:", prediccion_SVM)
        return prediccion_SVM


def prediccionCNN(imagen):
    # ImagenAplanada
    CNN_Predict = cargarNN("Recursos/CNNWeb")
    prediccion_CNN = CNN_Predict.predict(imagen.reshape(1, 32, 32, 3))
    print("Predicción CNM:", prediccion_CNN)
    predicciones = prediccion_CNN.flatten()
    clase = np.argmax(predicciones)
    valor = predicciones[clase]

    resultado = {'clase': clase, 'valor': valor}
    return resultado








