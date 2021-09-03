import kivy
import json
from cv2 import cv2
import numpy as np
import os
from pathlib import Path

class Function:
    def __init__(self, nombre, funcion, group):
        self.nombre = nombre # nombre asignado de la funcion
        self.funcion = funcion # codigo cv2 para esa funcion
        self.group = group # grupo al que pertenece la funcion
    
    def evaluate(self, inputs):
        try:
            outputs = eval (self.funcion)(*inputs)
        except cv2.error:
            print(cv2.error)
        return outputs

def BuscarFuncion(nombre):
    for element in funciones:
        if element.nombre == nombre:
            return element

funciones = []
# creación de todas las funciones disponibles
dir = str(Path(__file__).parent.absolute())
with os.scandir(dir + '\\funciones') as json_files:
    for element in json_files:
        json_file = open(element)
        data = json.load(json_file)
        for key in data:
            try:
                #BuscarFuncionJson(key['nombre'])
                fun = Function(key['nombre'], key['funcion'], key['group'])
                funciones.append(fun)
            except KeyError:
                pass


def orderby_groups():
    groups = {}
    for element in funciones:
        if element.group in groups:
            groups[element.group].append(element)
        else:
            groups[element.group] = [element]
    return groups
