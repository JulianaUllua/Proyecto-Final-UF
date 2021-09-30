#import:
import kivy
import os
from pathlib import Path
from kivy.core import window
from kivy.uix.behaviors.button import ButtonBehavior
from kivy_garden.contextmenu.context_menu import ContextMenuDivider
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
from cv2 import cv2
from numpy.lib.type_check import imag
from toposort import toposort
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from functools import partial

import kivy.garden 
from kivy.garden.matplotlib import FigureCanvasKivyAgg
from kivy.app import App
from kivy.config import Config
import kivy.properties as kprop

from kivy.lang import Builder
from kivy.factory import Factory

import kivy_garden.contextmenu
from kivy_garden.contextmenu import AbstractMenuItem

#import widgets:
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scatter import Scatter
from kivy.uix.scatterlayout import ScatterLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.slider import Slider
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget

from kivy.uix.label import Label
from kivy.uix.bubble import Bubble
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelHeader

from kivy.core.window import Window
from kivy.properties import ObjectProperty
from kivy.graphics.texture import Texture
from kivy.graphics import Color, Ellipse, Line, Rectangle, Bezier 
import kivy.graphics.instructions as kins

#import modulo clase
import Class_Function as CFunction
import Class_MyScatterLayout as CScatter


# pylint: disable=no-member

Config.set('input', 'mouse', 'mouse,disable_multitouch')

class MyApp(App):
    
    def build(self):
        sm = ScreenManager()
        mf = MainFloatScreen(name= 'main screen')
        mf.add_widget(MainFloatLayout())
        sm.add_widget(mf)
        sm.add_widget(SavedFloatScreen(name = "saved screen"))

        sm.add_widget(MenuScreen(name='menu screen'))
        
        sm.current = 'menu screen'

        return sm
        
    
class Pipeline:
    
    def __init__(self, bloque_load):
        self.pipe = []
    # bloque_load: objeto de clase Bloque con Function = Load porque se crea solo cuando hay una nueva imagen cargada  
        try:
            if bloque_load.funcion.nombre != "Load Image":
                raise NameError
            else:
                self.pipe.append(bloque_load)
        except NameError: 
            print("NameError: El primer bloque debe ser Load Image")

    def reset(self):
        for scatter in self.pipe[1:]:
            scatter.inputs.clear()
        self.pipe = []

    # para append de los proximos bloques: Pipeline.pipe.append()

    def imshow(self):
        try:
            if isinstance(self.pipe[-1].outputs.values(), np.ndarray):
                cv2.imshow('Imagen resultado del pipe', self.pipe[-1].outputs.values())
        except TypeError:
            for element in self.pipe[-1].outputs.values():
                if isinstance(element, np.ndarray):
                    cv2.imshow('Imagen resultado del pipe', element)
        cv2.waitKey(0)

    def output_toinput(self, node, line = 0):
        # recibe un node para que se asigne su output al input del siguiente elemento en el pipe
        # recibe line != 0 si hay union de tipo output-parametro input
        # --
        if node in self.pipe: #se puede sacar
            i = self.pipe.index(node)
            for key, value in self.pipe[i].outputs.items():
                if key == "retval" or "dst": # si es imagen es el output como antes
                    try:
                        if line != 0:
                            if key == line.button_output.parameter_text: #union de tipo output-parametro input
                                line.scat_inp.parameters[line.button_input.parameter_text] = self.pipe[i].outputs[line.button_output.parameter_text]
                                #ans = "eval({} = {})".format(line.button_input.parameter_text, self.pipe[i].outputs[line.button_output.parameter_text])
                                #line.scat_inp.parameters[line.button_input.parameter_text] = ans
                                #print(ans)
                        elif len(self.pipe[i+1].inputs) >= self.pipe[i+1].in_images: # solo acepta el numero de imagenes que se determina en el json como entrada. sino se limpia la lista y se agregan de cero
                            self.pipe[i+1].inputs.clear()
                            self.pipe[i+1].inputs.append(value)
                        else:
                            self.pipe[i+1].inputs.append(value)
                        
                        # node a evaluar deberá usar luminance si: es del tipo Change Colorspace o el node anterior en pipe usa luminance
                        #if self.pipe[i+1].funcion.nombre == 'Change Colorspace' or (self.pipe[i].colorfmt == 'luminance'and self.pipe[i+1].in_images<2):
                        #    self.pipe[i+1].colorfmt = 'luminance'           
                        
                    except IndexError:
                        pass              


class MainFloatLayout(FloatLayout):
    myscatter_aux = 0 # auxiliar para guardar scatter_output
    button_output_aux = 0 # auxiliar para guardar button_output
    mypos_aux = 0 # auxiliar para guardar 
    scats = [] # list de scatter_ids en orden
    lines_array = [] # list de scatter_pos en orden
    lines_list = [] # list de objetos Myline
    scatter_count = 0 # para asignar id a scatter
    scatter_list = [] # guarda objetos scatter, vertices
    scatter_graph = {} # diccionario de grafo: {['key]:['value','value']}, key=scatter_id del output, value=scatter_id de los inputs con los que esté conectado
    #scatter_graph_string = [] # lista de grafo con type=string
    list_toposort = [] #lista de graph en orden de ejecucion 
    list_pipelines = [] #lista de pipelines creados
    start_blocks = [] # guarda objetos scatter del tipo Load Image, para comenzar los paths desde estos
    location = (Window.system_size[0]*0.10)

    def __init__(self, **kwargs):
        super(MainFloatLayout, self).__init__(**kwargs)
        
    def new_bloque(self, value, scat_id = "new"):
        Fun = CFunction.search_function(value)
        #if scat_id == 0:
         #   scat_id = str(self.scatter_count)
        if scat_id != "new":
            while (int(scat_id) != self.scatter_count):
                self.scatter_list.append(None)
                self.scatter_count +=1

        scatter = CScatter.MyScatterLayout(draw_line_pipe = self.draw_line_pipe, update_line = self.update_line, delete_scatter = self.delete_scatter, 
                funcion = Fun, scatter_id = str(self.scatter_count),size=(150, 150) ,  size_hint=(None, None), pos=(self.location + 100,(Window.system_size[1]/2)))
        if self.location < Window.system_size[0]* 0.8:
            self.location = self.location + 165
        else:
            self.location= Window.system_size[0]*0.10
        self.ids.bloques_box.add_widget(scatter)
        self.scatter_list.append(scatter)
        self.scatter_count +=1

        if value == "Load Image":
            self.start_blocks.append(scatter.scatter_id) # para comenzar los paths desde estos
            
            #filename = r'C:\Users\trini\Pictures\lena.png'
            filename = r'C:\Users\Juliana\Pictures\coins.jpg'
            #filename = r'C:\Users\Juliana\Downloads\18_08_21\coins.jpg'
            scatter.inputs.append(filename)
        return scatter

    def draw_line_pipe(self, myscatter, button_id, instance):
        if instance.last_touch.button =='right':
            for line in self.lines_list:
                if instance == line.button_output or instance == line.button_input:
                    self.popup_delete_line(line)
        else:
            pos = instance.pos # posicion del boton

            if button_id == "outputs":
                self.myscatter_aux = myscatter #guardo myscatter para después unirlo
                self.button_output_aux = instance
                mypos = [myscatter.pos[0] + pos[0] + instance.size[0], myscatter.pos[1] + pos[1] + instance.size[1]/2] #suma posicion del myscatter + posicion del button
                self.line_flag = False
                for line in self.lines_list: #revisa las lineas existentes para que no permitir que se hagan dos sobre el mismo boton
                    if line.button_output == instance:
                        self.line_flag = True
                        break

                if not self.line_flag:
                    if myscatter.scatter_id in self.scats:
                        i = self.scats.index(myscatter.scatter_id) 
                        self.scats.remove(myscatter.scatter_id)
                        self.lines_array.pop(i)
                    self.scats.append(myscatter.scatter_id) #se lo agrega a las listas en nueva posicion
                    i = self.scats.index(myscatter.scatter_id)
                    self.set_list(i,mypos, self.lines_array) #equivalente a self.lines_array.append(mypos)

                    if myscatter.scatter_id not in self.scatter_graph:
                        #self.set_list(myscatter.scatter_id,{},self.scatter_graph) #equivalente a Function: Add Vertex
                        self.scatter_graph[myscatter.scatter_id] = set()

            elif button_id == "inputs" or "input_parameter":
                if self.myscatter_aux != 0: #si hay un scatter para unir
                    if self.line_flag:
                        for line in self.lines_list:
                            if line.button_output == self.button_output_aux:
                                for key, value in self.scatter_graph.items():
                                    if key == line.scatter_output:
                                        value_to_pop = line.scatter_input
                                        self.scatter_graph[line.scatter_output].remove(value_to_pop)
                                        break
                                self.lines_list.remove(line)
                                mypos1 = line.points[0]
                                line.clear_lines()
                                del line
                                break
                        
                    mypos = [myscatter.pos[0] + pos[0], myscatter.pos[1] + pos[1] + instance.size[1]/2]
                    
                    if myscatter.scatter_id in self.scats:
                        i = self.scats.index(myscatter.scatter_id)                    
                        self.scats.remove(myscatter.scatter_id)
                        self.lines_array.pop(i)
                    self.scats.append(myscatter.scatter_id)
                    i = self.scats.index(myscatter.scatter_id)
                    self.set_list(i,mypos, self.lines_array)
                    
                    #points = [mypos1, mypos]

                    if self.line_flag:
                        points = [mypos1, mypos]
                    else:
                        points = [self.lines_array[-2], self.lines_array[-1]]
                    myline = MyLine(self.myscatter_aux.scatter_id, myscatter.scatter_id, self.button_output_aux, instance, points, myscatter)
                    self.lines_list.append(myline) #equivalente a self.set_list(i, myline, self.lines_list)
                    self.ids.bloques_box.canvas.add(myline.line)
                    #equivalente a Function: Add Edge

                    #if button_id == "inputs":
                    if self.myscatter_aux.scatter_id not in self.scatter_graph:
                        self.scatter_graph[self.myscatter_aux.scatter_id] = set()
                    self.scatter_graph[self.myscatter_aux.scatter_id].add(myscatter.scatter_id)
                        #print(self.scatter_graph)
                        #self.set_list(self.myscatter_aux.scatter_id,[myscatter.scatter_id],self.scatter_graph) #ver. supone que solo un scatter va a estar conectado. no agrega relacion inversa

                    self.myscatter_aux = 0   

    def set_list(self, i, v, l):
      try:
          l[i] = v
      except IndexError:
          for _ in range(i-len(self.lines_array)+1):
              l.append([])
          l[i] = v

    def update_line(self, myscatter):
        if myscatter.scatter_id in self.scats:
            for myline in self.lines_list:
                i = self.scats.index(myscatter.scatter_id)
                if myline.scatter_input == myscatter.scatter_id:
                    mypos = [myscatter.pos[0] + myline.button_input.pos[0], myscatter.pos[1] + myline.button_input.pos[1] + myline.button_input.size[1]/2]
                    myline.points[1] = mypos
                    self.set_list(i, mypos, self.lines_array)
                    myline.update_line(myline.points)
                elif myline.scatter_output == myscatter.scatter_id:
                    mypos = [myscatter.pos[0] + myline.button_output.pos[0] + myline.button_output.size[0], myscatter.pos[1] + myline.button_output.pos[1] + myline.button_output.size[1]/2]
                    myline.points[0] = mypos
                    self.set_list(i, mypos, self.lines_array)
                    myline.update_line(myline.points)       
    
    def find_pipes(self):
        for pipe in self.list_pipelines:
            pipe.reset()
            del pipe
        
        self.list_pipelines = []
        # se ordena el graph en orden según dependencias
        self.list_toposort = list(toposort(self.scatter_graph))
        
        self.list_toposort.reverse() # lista guarda en orden en que hay que ejecutar los bloques
        print("toposort:", str(self.list_toposort))
        
        # se guardan los primeros y los ultimos en la lista para encontrar los paths
        start_blocks = [block for block in self.start_blocks if block in self.scats]
        try:
            finish_blocks = self.list_toposort[-1]
        except IndexError: #en caso de que haya un solo bloque, no funciona usar [-1], y da index error. No hay que crear ningun path, asique simplemente es un pass
            pass

        # se buscan los path
        for start in start_blocks:
            for finish in finish_blocks:
                paths = self.find_all_paths(self.scatter_graph, start, finish)
                if paths != None:
                    for p in paths:
                        if p != None:
                            # se crea pipe a partir del path
                            self.create_pipe(p) #ver. se puede hacer un constructor de la clase Pipeline
                            print("paths:",p)
                else:
                    print("Path no encontrado")
        
        self.run_pipes()

    def run_pipes(self): 
        for group in self.list_toposort:
            group = list(group) # se guarda al set como list
            if len(group) == 1: #hay un solo elemento en el grupo, no trabajo en paralelo
                self.scatter_list[int(group[0])].evaluate()
            else: 
                #hay varios elementos en el grupo, se ejecutan todos juntos con Multiprocessing                
                scats = [self.scatter_list[int(node)] for node in group]
                with ThreadPoolExecutor(max_workers=None) as executor:
                    results = executor.map(CScatter.MyScatterLayout.evaluate, scats)
                    i = 0
                    for values in results:
                        if isinstance(values, list):
                            for element in values:
                                if isinstance(element, np.ndarray):
                                    CScatter.MyScatterLayout.view_scatter_image(scats[i], element)
                                    print(type(element))
                        i=i+1

            # asignacion de outputs a inputs            
            for node in group:
                for pipeline in self.list_pipelines:
                    if self.scatter_list[int(node)] in pipeline.pipe:
                        for line in self.lines_list:
                            if line.scatter_output == node:
                                if isinstance(line.button_input, CScatter.MyParameterButton) and line.button_input.button_id == 'input_parameter':
                                    pipeline.output_toinput(self.scatter_list[int(node)], line)
                                elif group != list(self.list_toposort[-1]):
                                    pipeline.output_toinput(self.scatter_list[int(node)])
                
    def find_all_paths(self, graph, start, end, path=[]):
        path = path + [start]
        if start == end:
            return [path]
        if start not in graph:
            return []
        paths = []
        for node in graph[start]:
            if node not in path:
                newpaths = self.find_all_paths(graph, node, end, path)
                for newpath in newpaths:
                    paths.append(newpath)
        return paths

    def create_pipe(self, path): 
        # se construye objeto Pipeline a partir del primer bloque. 
        try:
            if self.scatter_list[int(path[0])].funcion.nombre != "Load Image":
                raise NameError
            else:
                pipeline = Pipeline(self.scatter_list[int(path[0])])

                # se agregan los demás bloques al pipeline en el orden marcado por path
                for scatter in path[1:]:
                    pipeline.pipe.append(self.scatter_list[int(scatter)])
                
                # se agrega pipeline a lista de pipelines existentes
                self.list_pipelines.append(pipeline)
        except NameError:
            print("Warning: Hay uno o mas pipelines sin unir al bloque Load Image")
        
    def delete_scatter(self, myscatter):
        self.ids.bloques_box.remove_widget(myscatter)
        self.scatter_list[int(myscatter.scatter_id)] = None # ese lugar en la lista queda vacio. no se remueve para no perder indices = ids
        if myscatter.scatter_id in self.scats:
            self.scats.remove(myscatter.scatter_id)
            for myline in self.lines_list: # Eliminamos la union (myline)
                if myline.scatter_input == myscatter.scatter_id or myline.scatter_output == myscatter.scatter_id:
                    myline.clear_lines()
                    self.lines_list[self.lines_list.index(myline)] = None
                    del myline
            self.lines_list = list(filter(None, self.lines_list))
            for key in self.scatter_graph:
                if myscatter.scatter_id in self.scatter_graph[key]:
                    self.scatter_graph[key].discard(myscatter.scatter_id)
            
            self.scatter_graph = {key:val for key, val in self.scatter_graph.items() if key != myscatter.scatter_id}

    def clear_screen(self):
        self.ids.bloques_box.clear_widgets()
        self.scatter_list.clear()
        self.scatter_count = 0
        self.scats.clear()
        for myline in self.lines_list:
            myline.clear_lines()
        self.lines_list.clear()
        self.lines_array.clear()
        self.scatter_graph.clear()
    
    def popup_delete_line(self, line):
        show = Popup_Delete_Line(self, line)
        self.delete_line_popup = Popup(title="Delete Line", content=show,size_hint=(None,None),size=(400,150))
        self.delete_line_popup.open()

    def delete_line(self, line):
        self.lines_list.remove(line)
        line.clear_lines()
        self.delete_line_popup.dismiss()

        for key in self.scatter_graph:
            if line.scatter_output == key:                
                if line.scatter_input in self.scatter_graph[key]:
                    self.scatter_graph[key].discard(line.scatter_input)
                    break
        if self.scatter_graph[key] == set():
            del self.scatter_graph[key]             

    """def Extract_Pipe_Code(self):
        i=0
        dir = str(Path(__file__).parent.absolute())
        with open(dir + '\\autogenerated_code.py','w+') as secondfile:
            importing = 'import numpy as np\nfrom cv2 import cv2\nfrom matplotlib import pyplot as plt\n\n'
            secondfile.write(str(importing))
            n=0
            for pipeline in self.list_pipelines:
                n += 1
                if pipeline.pipe[0].funcion.nombre == "Load Image":
                    secondfile.write("\nfilename_{} = r'{}'".format(i,pipeline.pipe[0].inputs[0]))
                    secondfile.write("\nimg_{} = ".format(n) + pipeline.pipe[0].funcion.funcion  + "(filename_" + str(i) + ")")
                    i+=1
                    
                if len(pipeline.pipe)>1:
                    for fun in pipeline.pipe[1:]:
                        parameters_aux = "img_{}".format(n)
                        for x in range (n+1, fun.in_images + 1):
                            parameters_aux = parameters_aux + ", img_{}".format(n)
                            
                        for value in fun.parameters.values():
                            parameters_aux = parameters_aux+ "," + str(value) 
                        n += 1
                        secondfile.write("\nimg_{} = {}".format(n,fun.funcion.funcion + "(" + parameters_aux +")"))
                    
                secondfile.write("\ncv2.imshow('Imagen Resultado', img_{})\ncv2.waitKey(0)".format(n))  
                secondfile.write("\n")"""
          
            
    def Extract_Pipe_Code(self):    
        dir = str(Path(__file__).parent.absolute())
        with open(dir + '\\autogenerated_code.py','w+') as secondfile:
            in_imag = [1] * self.scatter_count
            p_aux = []
            for x in range (0, self.scatter_count):
                p_aux.append("img_{}".format(x-1))
            s_c = 0
            importing = 'import numpy as np\nfrom cv2 import cv2\nfrom matplotlib import pyplot as plt\n\n'
            secondfile.write(str(importing))
            for group in self.list_toposort:
                group = list(group) # se guarda al set como list                            
                for node in group:                    
                    for pipeline in self.list_pipelines:
                        if self.scatter_list[int(node)] in pipeline.pipe:
                            i = pipeline.pipe.index(self.scatter_list[int(node)])
                        
                            if self.scatter_list[int(node)].funcion.nombre == "Load Image":    
                                p = pipeline.pipe[i].scatter_id                              
                                secondfile.write("\nfilename_{} = r'{}'".format(p,self.scatter_list[int(node)].inputs[0]))
                                secondfile.write("\nimg_{} = ".format(p) + self.scatter_list[int(node)].funcion.funcion  + "(filename_" + str(p) + ")\n")

                            else:
                                p = pipeline.pipe[i-1].scatter_id
                                parameters_aux = "img_{}".format(p)
                                
                                if self.scatter_list[int(node)].in_images > in_imag[int(node)]:
                                    p_aux[int(node)] = p_aux[int(node)] + ", img_{}".format(p)
                                    in_imag[int(node)] += 1
                                    
                                else:
                                    n = str(int(s_c) - 1)
                                    idx = p_aux.index("img_" + n)
                                    if p_aux[int(node)] != "img_{}".format(idx):
                                        parameters_aux = p_aux[int(node)]
                                    for value in self.scatter_list[int(node)].parameters.values():
                                        if value != 'no input':
                                            parameters_aux = parameters_aux + "," + str(value) 
      
                                    secondfile.write("\nimg_{} = {}".format(pipeline.pipe[i].scatter_id , self.scatter_list[int(node)].funcion.funcion + "(" + parameters_aux +")\n"))
                s_c = s_c + 1
            try:
                finish_blocks = self.list_toposort[-1]
                for f in finish_blocks:
                    secondfile.write("\ncv2.imshow('Imagen Resultado', img_{})\ncv2.waitKey(0)".format(f))  
                    secondfile.write("\n")
            except IndexError: 
                pass            

    def show_extraer_popup(self, s = ""):
        show = Popup_Extraer_Codigo(self)
        if s != "save":  
            self.extraer_popup = Popup(title="Extraer Codigo", content=show,size_hint=(None,None),size=(400,150))
        else:
            box = GridLayout(rows = 2, row_force_default=True, row_default_height=80)
            button = Save_workspace_button(self)
            box.add_widget(button)
            self.extraer_popup = Popup(title="Save Pipeline As", content=box,size_hint=(None,None),size=(400,150))
        self.extraer_popup.open()

    def dismiss_extraer_popup(self):
        self.extraer_popup.dismiss()

    def image_viewer(self):
        iv = ImageViewer()
        for scat in self.scatter_list:
            if scat != None:
                th = TabbedPanelHeader(text='%d: %s' % (self.scatter_list.index(scat)+1, scat.funcion.nombre))
                th.width = th.texture_size[0]
                th.padding = 30,0
                th.font_size = '12sp'
                try:
                    if isinstance(scat.outputs.values(), np.ndarray):
                        image = scat.outputs.values()
                    else:
                        for element in scat.outputs.values():
                            if isinstance(element, np.ndarray):
                                image = element
                    
                    texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt= scat.colorfmt)
                    texture.blit_buffer(image.tobytes(order=None), colorfmt= scat.colorfmt, bufferfmt='ubyte')
                    texture.flip_vertical()

                    text = ("[b]Statistics[/b]" + '\nDimension: {}'.format(image.ndim) + '\nShape: {}'.format(image.shape) + 
                            '\nHeight: {}'.format(image.shape[0]) + '\nWidth: {}'.format(image.shape[1])) 
                    mywidget = MyWidget(text)
                    mywidget.ids.view_image.color = (1,1,1,1)
                    mywidget.ids.view_image.texture = texture

                    th.content = mywidget

                    if scat.colorfmt == 'bgr':
                        scat.myhist = MyHistogram(image)
                        mywidget.ids.histogram.add_widget(scat.myhist)
                    #plt.show()  
                    
                except TypeError:
                    pass

                iv.add_widget(th)

        self.popup = Popup(title='Image Viewer', content=iv, size_hint=(.9, .9), size=Window.size)
        self.popup.open()

    def run_pipes_until(self, scatter_id):
        if  self.list_toposort != [] and scatter_id in self.list_toposort:
            for group in self.list_toposort:
                if scatter_id in group:
                    index = self.list_toposort.index(group)
                    break
                else:
                    index = 0

            for group in self.list_toposort[index:]:
                group = list(group) # se guarda al set como list
                if len(group) == 1: #hay un solo elemento en el grupo, no trabajo en paralelo
                    self.scatter_list[int(group[0])].evaluate()
                else: #hay varios elementos en el grupo, se ejecutan todos juntos con Multiprocessing                
                    scats = [self.scatter_list[int(node)] for node in group]
                    with ThreadPoolExecutor(max_workers=None) as executor:
                        results = executor.map(CScatter.MyScatterLayout.evaluate, scats)
                        i = 0
                        for values in results:
                            if isinstance(values, list):
                                for element in values:
                                    if isinstance(element, np.ndarray):
                                        CScatter.MyScatterLayout.view_scatter_image(scats[i], element)
                            i=i+1

                # asignacion de outputs a inputs
                for node in group:
                    for pipeline in self.list_pipelines:
                        if self.scatter_list[int(node)] in pipeline.pipe:
                            for line in self.lines_list:
                                if line.scatter_output == node:
                                    if isinstance(line.button_input, CScatter.MyParameterButton) and line.button_input.button_id == 'input_parameter':
                                        pipeline.output_toinput(self.scatter_list[int(node)], line)
                                    elif group != list(self.list_toposort[-1]):
                                        pipeline.output_toinput(self.scatter_list[int(node)])

    def show_from_file_popup(self):
        dropdown = DropDown(size_hint_y = 1, size_hint_x =1) 
        dir = str(Path(__file__).parent.absolute().joinpath('saved_pipelines')) 
        with os.scandir(dir) as json_files:
            for element in json_files:
                button = Button(text = str(element.name), size_hint_y = None, height = 40)
                buttoncallbackin = partial(self.from_file, str(element.name))
                button.bind(on_press = buttoncallbackin )
                dropdown.add_widget(button)

        mainbutton = Button(text='Select workspace', size_hint=(500, 150))
        mainbutton.bind(on_release=dropdown.open)
        dropdown.bind(on_select=lambda instance, x: setattr(mainbutton, 'text', x))
        self.from_file_popup = Popup(title="Open workspace", content=dropdown,size_hint=(None,None), size = (500,200))
        self.from_file_popup.open()

    def from_file(self, filename, *args):
        self.from_file_popup.dismiss()
        dir = str(Path(__file__).parent.absolute())
        with open(dir + "\\saved_pipelines\\%s" %filename) as f:
            data = json.load(f)
            scatter_g = (data['scatter_graph'])
            self.scatter_graph = {}
            for key, value in scatter_g.items():
                self.scatter_graph[key] = set(value)
            #self.start_blocks = data['start_blocks']
            self.scats = data['scats']
            
            for item in data["scatter_list"]:
                if item is not None:
                    scatter = self.new_bloque(item['scatter']['nombre'],item['scatter']['scatter_id'])
                    scatter.parameters = (item['scatter']['parameters'])
                    #ver otra forma de arreglar excepcion
                    if item['scatter']['nombre'] == "Gaussian Blur":
                        scatter.parameters["ksize"] = (scatter.parameters["ksize"][0], scatter.parameters["ksize"][1])
                    scatter.pos = item['scatter']['pos']
                    for button in scatter.ids.inputs.children:
                        if isinstance(button, CScatter.MyParameterButton):
                            input = button
                        if isinstance(button, CScatter.MyIconButton):
                            input = button
                    for button in scatter.ids.outputs.children:
                        if isinstance(button, CScatter.MyParameterButton):
                            output = button

            for item in data["lines_list"]:
                myline = MyLine(item['line']['scatter_output'], item['line']['scatter_input'], output, input, item['line']['points'], scatter)
                self.lines_list.append(myline) 
                self.ids.bloques_box.canvas.add(myline.line)
        
        self.find_pipes()            

    def to_file(self, filename):
        self.extraer_popup.dismiss()
        try:
            dir = str(Path(__file__).parent.absolute())
            with open(dir + '\\saved_pipelines\\%s.json' % filename, 'w') as f:
                    json_data = {
                        'scatter_graph' : self.scatter_graph,
                        'start_blocks' : self.start_blocks,
                        'scats' : self.scats,
                        'scatter_list' : self.scatter_list,
                        'lines_list' : self.lines_list
                    }
                    encoder = MultipleJsonEncoders(SetEncoder, MyLineEncoder, MyScatterLayoutEncoder, NumpyArrayEncoder)
                    json.dump(json_data, f, indent=4, ensure_ascii=False, cls=encoder)

        except Exception as ex:
            print(ex)

class MultipleJsonEncoders():
    #Combine multiple JSON encoders

    def __init__(self, *encoders):
        self.encoders = encoders
        self.args = ()
        self.kwargs = {}

    def default(self, obj):
        for encoder in self.encoders:
            try:
                return encoder(*self.args, **self.kwargs).default(obj)
            except TypeError:
                pass
        raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        enc = json.JSONEncoder(*args, **kwargs)
        enc.default = self.default
        return enc

class SetEncoder(json.JSONEncoder):
    def default(self, obj):
       if isinstance(obj, set):
           return list(obj)

       return json.JSONEncoder.default(self, obj)

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class MyScatterLayoutEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, CScatter.MyScatterLayout): # this is the item class you want to serialize
            try:
                scatter_dict = defaultdict(dict)
                scatter_dict['nombre'] = obj.funcion.nombre
                scatter_dict['funcion'] = obj.funcion.funcion
                scatter_dict['group'] = obj.funcion.group

                # por ahora decidi solo guardar el filename de los load image porq los arrays generados desp ocupan mucho espacio y podemos generarlos en output to input devuelta
                if obj.funcion.nombre == "Load Image":
                    scatter_dict['inputs'] = obj.inputs
                scatter_dict['parameters'] = obj.parameters
                scatter_dict['scatter_id'] = obj.scatter_id
                scatter_dict['pos'] = obj.pos

                scatter_dict['in_images'] = obj.in_images
                scatter_dict['out_images'] = obj.out_images
                
                return {'scatter': scatter_dict} 
            except TypeError:
                pass
        return json.JSONEncoder.default(self, obj)

class MyLineEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, MyLine): # this is the item class you want to serialize
            try:
                property_dict = defaultdict(dict)
                property_dict['scatter_output'] = obj.scatter_output
                property_dict['scatter_input'] = obj.scatter_input

                property_dict['points'] = obj.points

                scatter_dict = defaultdict(dict)
                scatter_dict['nombre'] = obj.scat_inp.funcion.nombre
                scatter_dict['funcion'] = obj.scat_inp.funcion.funcion
                scatter_dict['group'] = obj.scat_inp.funcion.group

                property_dict['scat_inp']['funcion']= scatter_dict

                
                property_dict['scat_inp']['parameters'] = obj.scat_inp.parameters
                property_dict['scat_inp']['in_images'] = obj.scat_inp.in_images
                property_dict['scat_inp']['out_images'] = obj.scat_inp.out_images
                
                return {'line': property_dict} 
            except TypeError:
                pass
        return json.JSONEncoder.default(self, obj)


class MainFloatScreen(Screen):
    pass

class SavedFloatScreen(Screen):
    def fileload(self):
        mf = MainFloatLayout()
        self.add_widget(mf)
        mf.show_from_file_popup()
    pass

class MenuScreen(Screen):
    pass

class MyHistogram(FigureCanvasKivyAgg):
    def __init__(self, image, **kwargs):
        super(MyHistogram,self).__init__(plt.gcf(), **kwargs)
        color = ('b','g','r')
        for i,col in enumerate(color):
            self.histr = cv2.calcHist([image],[i],None,[256],[0,256])
            plt.plot(self.histr,color = col)
            plt.xlim([0,256])
            
class ImageViewer(TabbedPanel):
    pass

class MyWidget(BoxLayout):
    text = kprop.StringProperty() #default value shown

    def __init__(self, text, **kwargs):
        super(MyWidget,self).__init__(**kwargs)
        self.text = text
    pass

class Popup_Extraer_Codigo(FloatLayout):
    def __init__(self, floatlayout, **kwargs):
        super(Popup_Extraer_Codigo, self).__init__(**kwargs)
        self.mainfloat = floatlayout
    pass

class Popup_Delete_Line(FloatLayout):
    def __init__(self, floatlayout, line, **kwargs):
        super(Popup_Delete_Line, self).__init__(**kwargs)
        self.mainfloat = floatlayout
        self.line = line
    pass

class CodeBlock():
    #Para la creacion de codigo. Por ahora no la use
    def __init__(self, head, block):
        self.head = head
        self.block = block
    def __str__(self, indent=""):
        result = indent + self.head + ":\n"
        indent += "    "
        for block in self.block:
            if isinstance(block, CodeBlock):
                result += block.__str__(indent) 
            else:
                result += indent + block + "\n" 
        return result

class MyLine:

    def __init__(self, scatter_output, scatter_input, button_output, button_input, points, scat_inp = 0):
        self.scatter_output = scatter_output
        self.scatter_input = scatter_input #id
        self.button_output = button_output
        self.button_input = button_input
        self.line = kins.InstructionGroup()
        self.points = points
        self.line.add(Line(points=points, width=1))
        self.line.add(Ellipse(pos=(points[0][0]-6, points[0][1]-5), size=(7,7)))
        self.line.add(Ellipse(pos=(points[1][0]-6, points[1][1]-5), size=(7,7)))
        self.scat_inp = scat_inp #MyScatterLayout

    def update_line(self, points):
        self.line.clear()
        self.line.add(Line(points=points, width=1))
        self.line.add(Ellipse(pos=(points[0][0]-6, points[0][1]-5), size=(7,7)))
        self.line.add(Ellipse(pos=(points[1][0]-6, points[1][1]-5), size=(7,7)))
    
    def clear_lines(self):
        self.line.clear()

class FilterDD(Factory.DropDown):
    ignore_case = Factory.BooleanProperty(True)
    options = Factory.ListProperty()
    options_groups = Factory.ListProperty()

    def __init__(self, dismiss_on_select, **kwargs):
        self._needle = None
        self._order = []
        self._widgets = {}
        self.dismiss_on_select = dismiss_on_select
        super(FilterDD, self).__init__(**kwargs)
        groups = CFunction.orderby_groups()
                
        for key, value in groups.items():
            self.options_groups.append(key)
            for element in value:
                self.options.append(element.nombre)

    def on_options_groups(self, instance, values):
        _order = self._order
        _widgets = self._widgets
        for txt in values:
            if txt not in _widgets:
                _widgets[txt] = btn = MyLabel(text=txt, size_hint_y=None, height= 30)
                _order.append(txt)
    
    def on_options(self, instance, values):
        _order = self._order
        _widgets = self._widgets
        changed = False
        for txt in values:
            if txt not in _widgets:
                _widgets[txt] = btn = Factory.DDButton(text=txt)
                btn.bind()
                _order.append(txt)
                changed = True
        for txt in _order[:]:
            if txt not in values:
                #_order.remove(txt)
                #del _widgets[txt]
                changed = True
        if changed:
            self.apply_filter(self._needle)

    def apply_filter(self, needle):
        self._needle = needle
        self.clear_widgets()
        _widgets = self._widgets
        add_widget = self.add_widget
        ign = self.ignore_case
        _lcn = needle and needle.lower()
        for haystack in self._order:
            _lch = haystack.lower()
            if not needle or ((ign and _lcn in _lch) or 
                         (not ign and needle in haystack)):
                if haystack in self.options_groups:
                    index = self.options_groups.index(haystack)
                    i = self._order.index(haystack)
                    if index+1 != len(self.options_groups):
                        j = self._order.index(self.options_groups[index+1])
                        for element in self._order[i:j]:
                            add_widget(_widgets[element])
                    else:
                        for element in self._order[i:]:
                            add_widget(_widgets[element])
                elif _widgets[haystack] not in self.container.children:
                    add_widget(_widgets[haystack])  

class FilterDDTrigger(Factory.BoxLayout):
    def __init__(self, **kwargs):
        super(FilterDDTrigger, self).__init__(**kwargs)
        self._prev_dd = None
        self._textinput = ti = Factory.TextInput(multiline=False, hint_text='Enter function name', size_hint=(0.5,None), height=30, pos_hint={'center_x':0.5, 'center_y':0.5}, 
        background_color= (0.2, 0.2, 0.2, 1.0), foreground_color= (1,1,1,1), cursor_color= (1,1,1,1))
        ti.bind(text=self._apply_filter)
        ti.bind(on_text_validate=self._on_enter)
        self._button = btn = Factory.Button(text=self.text, background_normal = '', background_down = '', background_disabled_normal= "", background_disabled_down= "")
        btn.background_color= (0.2, 0.71, 0.9, 1) if btn.state == 'down' else (0.2, 0.2, 0.2, 1.0)
        btn.bind(on_release=self._on_release)
        self.add_widget(btn)

    text = Factory.StringProperty('Open')
    def on_text(self, instance, value):
        self._button.text = value

    dropdown = Factory.ObjectProperty(None, allownone=True)
    def on_dropdown(self, instance, value):
        _prev_dd = self._prev_dd
        if value is _prev_dd:
            return
        if _prev_dd:
            _prev_dd.unbind(on_dismiss=self._on_dismiss)
            _prev_dd.unbind(on_select=self._on_select)
        if value:
            value.bind(on_dismiss=self._on_dismiss)
            value.bind(on_select=self._on_select)
        self._prev_dd = value

    def _apply_filter(self, instance, text):
        if self.dropdown:
            self.dropdown.apply_filter(text)

    def _on_release(self, *largs):
        if not self.dropdown:
            return
        self.remove_widget(self._button)
        self.add_widget(self._textinput)
        self.dropdown.open(self)
        self._textinput.focus = True

    def _on_dismiss(self, *largs):
        self.remove_widget(self._textinput)
        self.add_widget(self._button)
        self._textinput.text = ''

    def _on_select(self, instance, value):
        self.parent.parent.parent.new_bloque(value)

    def _on_enter(self, *largs):
        container = self.dropdown.container
        if container.children:
            self.dropdown.select(container.children[-1].text)
        else:
            self.dropdown.dismiss()

class MyLabel(Label):
    def __init__(self, **kwargs):
        super(MyLabel, self).__init__(**kwargs)
        pass

class Save_workspace_button(BoxLayout):
    def __init__(self, floatlayout, **kwargs):
        super(Save_workspace_button, self).__init__(**kwargs)
        self.mainfloat = floatlayout
    pass

if __name__ == '__main__':
    MyApp().run()
