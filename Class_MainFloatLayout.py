#import:
import kivy
import os
from pathlib import Path
from kivy.core import window
from kivy_garden.contextmenu.context_menu import ContextMenuDivider
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
from cv2 import cv2
from numpy.lib.type_check import imag
from toposort import toposort
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, process, as_completed

matplotlib.use("module://kivy.garden.matplotlib.backend_kivy")
import kivy.garden 
from kivy.garden.matplotlib import FigureCanvasKivyAgg
from kivy.app import App
from kivy.config import Config
import kivy.properties as kprop
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelHeader
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
from kivy.core.window import Window
from kivy.uix.label import Label
from kivy.uix.bubble import Bubble
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.uix.image import Image

from kivy.core.image import Image
from kivy.graphics.texture import Texture
from kivy.graphics import Color, Ellipse, Line, Rectangle 
import kivy.graphics.instructions as kins

#import modulo clase
import Class_Function as CFunction
import Class_MyScatterLayout as CScatter


# pylint: disable=no-member

Config.set('input', 'mouse', 'mouse,disable_multitouch')

class MyApp(App):

    def build(self):
        return MainFloatLayout()
        
    
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
    scatter_graph_string = [] # lista de grafo con type=string
    list_toposort = [] #lista de graph en orden de ejecucion 
    list_pipelines = [] #lista de pipelines creados
    start_blocks = [] # guarda objetos scatter del tipo Load Image, para comenzar los paths desde estos
    location = (Window.system_size[0]*0.10)

    def __init__(self, **kwargs):
        super(MainFloatLayout, self).__init__(**kwargs)
        
    def new_bloque(self, obj):
        nombre = obj.text
        Fun = CFunction.BuscarFuncion(nombre)
        scatter = CScatter.MyScatterLayout(draw_line_pipe = self.draw_line_pipe, update_line = self.update_line, delete_scatter = self.delete_scatter, 
                funcion = Fun, size=(150, 150), scatter_id = str(self.scatter_count),  size_hint=(None, None), pos=(self.location,(Window.system_size[1]/2)))
        if self.location < Window.system_size[0]* 0.8:
            self.location = self.location + 150
        else:
            self.location= Window.system_size[0]*0.10
        self.ids.bloques_box.add_widget(scatter)
        self.scatter_list.append(scatter)
        self.scatter_count +=1

        if nombre == "Load Image":
            self.start_blocks.append(scatter.scatter_id) # para comenzar los paths desde estos
            
            #filename = r'C:\Users\trini\Pictures\lena.png'
            filename = r'C:\Users\Juliana\Pictures\cell.png'
            #filename = r'C:\Users\Juliana\Downloads\18_08_21\coins.jpg'
            scatter.inputs.append(filename)

    def draw_line_pipe(self, myscatter, button_id, instance):
        pos = instance.pos # posicion del boton

        if button_id == "outputs":
            self.myscatter_aux = myscatter #guardo myscatter para después unirlo
            self.button_output_aux = instance
            mypos = [myscatter.pos[0] + pos[0] + instance.size[0], myscatter.pos[1] + pos[1] + instance.size[1]/2] #suma posicion del myscatter + posicion del button
            self.line_flag = False
            for line in self.lines_list:
                if line.button_output == instance:
                    self.line_flag = True
                    break

            if not self.line_flag:
                if myscatter.scatter_id in self.scats: #ver. se lo remueve de las listas para ponerlo en una nueva posicion, pero en caso de ser una segunda unión no habria que eliminarlo
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
        
        #try:
        self.list_pipelines = []
        # se ordena el graph en orden según dependencias
        self.list_toposort = list(toposort(self.scatter_graph))
        self.list_toposort.reverse() # lista guarda en orden en que hay que ejecutar los bloques
        print("toposort:", str(self.list_toposort))
        
        # se guardan los primeros y los ultimos en la lista para encontrar los paths
        start_blocks = [block for block in self.start_blocks if block in self.scats]
        finish_blocks = self.list_toposort[-1]

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

        #except IndexError:
        #    print("IndexError: Debe unir por lo menos dos bloques")

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
    
    def Extract_Pipe_Code(self):
        i=0
        dir = str(Path(__file__).parent.absolute())
        #with open(r'C:\Users\trini\OneDrive\Favaloro\Tesis\Código\06-Junio\05_06_21\autogenerated_code.py','w+') as secondfile: 
        with open(dir + '\\autogenerated_code.py','w+') as secondfile:
            importing = 'import numpy as np\nfrom cv2 import cv2\nfrom matplotlib import pyplot as plt\n\n'
            secondfile.write(str(importing))
            for pipeline in self.list_pipelines:
                n=0
                if pipeline.pipe[0].funcion.nombre == "Load Image":
                    secondfile.write("\nfilename_{} = r'{}'".format(i,pipeline.pipe[0].inputs[0]))
                    secondfile.write("\nimg_{} = ".format(n) + pipeline.pipe[0].funcion.funcion  + "(filename_" + str(i) + ")")
                    i+=1
                for fun in pipeline.pipe[1:]:
                    parameters_aux = "img_{}".format(n)
                    for value in fun.parameters.values():
                        parameters_aux = parameters_aux+ "," + str(value) 
                    n += 1
                    secondfile.write("\nimg_{} = {}".format(n,fun.funcion.funcion + "(" + parameters_aux +")"))
                    
                secondfile.write("\ncv2.imshow('Imagen Resultado', img_{})\ncv2.waitKey(0)".format(n))  
                secondfile.write("\n")

    def show_extraer_popup(self):
        show = Popup_Extraer_Codigo(self)
        self.extraer_popup = Popup(title="Extraer Codigo", content=show,size_hint=(None,None),size=(400,150))
        self.extraer_popup.open()

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
                        myhist = MyHistogram(image)
                        mywidget.ids.histogram.add_widget(myhist)
                    #plt.show()  
                    
                except TypeError:
                    pass

                iv.add_widget(th)

        self.popup = Popup(title='Image Viewer', content=iv, size_hint=(.9, .9), size=Window.size)
        self.popup.open()

    def run_pipes_until(self, scatter_id):
        if  self.list_toposort != []:
            for group in self.list_toposort:
                if scatter_id in group:
                    index = self.list_toposort.index(group)
                    break

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

    def on_touch_down(self,touch):
        if touch.button == 'right':
            if not hasattr(self, 'bubble'):
                self.bubble = CScatter.MyBubble(pos=self.to_local(self.center_x - self.width*0.5,self.center_y))
                self.ids.funcion_scatter.add_widget(self.bubble)

if __name__ == '__main__':
    MyApp().run()
