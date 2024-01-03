# En forms.py
from django import forms

class ImagenForm(forms.Form):
    imagen = forms.ImageField()
    opcion = forms.ChoiceField(choices=(('svm', 'Maquina de soporte vectorial - SVM'), ('cnn', 'Red neuronal convolucional - CNN')), widget=forms.RadioSelect)

