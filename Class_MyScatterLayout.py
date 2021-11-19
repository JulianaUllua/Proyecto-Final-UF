import kivy
import json
from cv2 import cv2
import warnings
import numpy as np
import os
from pathlib import Path

from kivy.app import App
from kivy.uix.scatter import Scatter
from kivy.uix.scatterlayout import ScatterLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.checkbox import CheckBox
from kivy.uix.actionbar import ActionButton

from kivy.uix.popup import Popup
from kivy.uix.bubble import Bubble

import kivy.graphics.transformation as ktransf
from kivy.graphics import Rectangle
import kivy.properties as kprop

from kivy.uix.dropdown import DropDown
from kivy.uix.spinner import Spinner

from kivy.core.image import Image
from kivy.core.window import Window
from kivy.graphics.texture import Texture

from kivy.event import EventDispatcher
from kivy.clock import Clock
from functools import partial

import kivy_garden.contextmenu
from kivy_garden.contextmenu import ContextMenuTextItem
from kivy_garden.contextmenu import ContextMenuItem
# pylint: disable=no-member
# esta linea arregla poder llamar atributos de Bloque en MyScatter
warnings.simplefilter(action='ignore', category=FutureWarning)

#bloque

class Bloque:
    colorfmt = 'bgr'

    def __init__(self, funcion):
    # funcion: objeto clase Function
    # inputs: entradas de la funcion, en general la imagen. Excepto para Load, son las salidas del bloque anterior
    # parameters: resto de los inputs del metodo "evaluate" de la clase Function
    # outputs: imagen y/o valores de salida que seran la entrada del siguiente bloque del pipe
    # in_images = numero de imagenes que tiene de entrada
    # out_images = numero de imagenes que tiene de salida
        self.funcion = funcion
        self.inputs = []
        self.parameters = {}
        self.outputs = {}
        self.popup_bloque = Parameters_Popup(save_parameters = self.save_parameters, 
                            cancel_parameters = self.cancel_parameters, restore_parameters = self.restore_parameters, 
                            try_parameters = self.try_parameters, duplicate = self.duplicate, size = (400,400))
        self.in_images = 0 
        self.out_images = 0 
        self.search_widgets()

    def evaluate(self):
        try:
            if (self.inputs == []):
                raise NameError
              
            else:
                if self.parameters != {}:
                    parameters_aux = []

                    for n in self.inputs:
                        parameters_aux.append(n)
                    
                    for value in self.parameters.values():
                        if value != 'no input':
                            parameters_aux.append(value)

                    outputs = self.funcion.evaluate(parameters_aux)
                    self.view_scatter_image(outputs)
                    
                else:
                    if self.funcion.nombre == "Load Image":
                        outputs = np.array(self.funcion.evaluate(self.inputs))
                    else:
                        outputs = self.funcion.evaluate(self.inputs)

                if isinstance(outputs, np.ndarray):
                    for key in self.outputs:
                        self.outputs[key] = outputs
                        #cv2.imwrite(r'C:\Users\trini\Pictures\Masked image.png', outputs)
                        if len(outputs.shape) > 2: #rgb
                            self.view_scatter_image(self.outputs.values()) 
                        else: #grayscale
                            self.colorfmt = 'luminance'
                            self.view_scatter_image(self.outputs.values()) 
                else:
                    for element in outputs:
                        if isinstance(element, np.ndarray):
                            for key in self.outputs:
                                self.outputs[key] = element #outputs del evaluate se guardan en diccionario outputs, tienen mismo orden
                                #cv2.imwrite(r'C:\Users\trini\Pictures\Masked image.png', element)
                                if len(element.shape) > 2: #rgb
                                    self.view_scatter_image(self.outputs.values()) 
                                else: #grayscale
                                    self.colorfmt = 'luminance' 
                                    self.view_scatter_image(self.outputs.values()) 
                        
                return list(self.outputs.values())

        except NameError: 
            print("NameError: Hay un o mas bloque/s sin input/s. Verifique las uniones")

    def try_evaluate(self, parameters):
        try:
            if (self.inputs == []):
                raise NameError
            else:
                if parameters != {}:
                    parameters_aux = []

                    for n in self.inputs:
                        parameters_aux.append(n)
                    
                    for value in parameters.values():
                        if value != 'no input':
                            parameters_aux.append(value)
                    outputs_aux = self.funcion.evaluate(parameters_aux)
                    if len(outputs_aux.shape) > 2: #rgb
                        self.view_scatter_image(outputs_aux) 
                    else: #grayscale
                        self.colorfmt = 'luminance'   
                        self.view_scatter_image(outputs_aux)

                else:
                    if self.funcion.nombre == "Load Image":
                        outputs_aux = np.array(self.funcion.evaluate(self.inputs))
                    else:
                        outputs_aux = self.funcion.evaluate(self.inputs)

                #if self.funcion.nombre != "Load Image":
                #    self.view_popup_image(outputs_aux)

                self.view_popup_image(outputs_aux)   
                self.view_scatter_image(outputs_aux)  
                

        except NameError: 
            print("NameError: Hay un o mas bloque/s sin input/s. Verifique las uniones")

    def view_popup_image(self, outputs):
        #Muestro imagen en Popup
        if isinstance(outputs, np.ndarray):
            image = outputs
        else:
            for element in outputs:
                if isinstance(element, np.ndarray):
                    image = element
        
        # create a Texture the correct size and format for the image
        texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt= self.colorfmt)
        #texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt= 'luminance')
        #para mostrar imagen en blanco y negro: colorfmt='luminance'

        # copy the data into the texture
        texture.blit_buffer(image.tobytes(order=None), colorfmt= self.colorfmt, bufferfmt='ubyte')

        # flip the texture
        texture.flip_vertical()

        # actually put the texture in the kivy Image widget
        self.popup_bloque.ids.image_popup.color = (1,1,1,1)
        self.popup_bloque.ids.image_popup.texture = texture

    def add_input_buttons(self, checkbox, value):
        if value and checkbox.was_active == False: # checkbox activado, No permito crear el mismo boton mas de una vez
            dir = str(Path(__file__).parent.absolute())
            if checkbox.parameter_text == 'dst':
                source = dir + '\\icons\\image_icon1.png'
            elif checkbox.parameter_text == 'retval':
                source = dir + '\\icons\\retval.png'
            elif checkbox.parameter_text == 'Color' or checkbox.parameter_text =='ColorMap':
                source = dir + '\\icons\\color.png'
            elif checkbox.parameter_text == 'mask':
                source = dir + '\\icons\\mask.png'
            elif checkbox.parameter_text == 'dsize':
                source = dir + '\\icons\\size.png'
            elif checkbox.parameter_text == 'fx' or checkbox.parameter_text == 'fy':
                source = dir + '\\icons\\scale.png'
            elif checkbox.parameter_text == 'Interpolation':
                source = dir + '\\icons\\graph.png'
            elif checkbox.parameter_text == 'ksize' or checkbox.parameter_text == 'SigmaX' or checkbox.parameter_text == 'SigmaY'or checkbox.parameter_text == 'Threshold' or checkbox.parameter_text == 'MaxVal' or checkbox.parameter_text == 'adaptiveMethod' or checkbox.parameter_text == 'blockSize' or checkbox.parameter_text == 'C' or checkbox.parameter_text == 'apertureSize'or checkbox.parameter_text =='ddepth'or checkbox.parameter_text == 'scale' or checkbox.parameter_text =='delta':
                source = dir + '\\icons\\number.png'
            elif checkbox.parameter_text == 'borderType' or checkbox.parameter_text == 'Type':
                source = dir + '\\icons\\opencv.png'
            
            button_input = MyParameterButton(button_id = 'input_parameter', parameter_text = checkbox.parameter_text, source = source,background_color = [0, 0, 0, 0])
            self.ids.inputs.size_hint = .2, (.2 + .05*(2))
            self.ids.inputs.add_widget(button_input)
            buttoncallbackin = partial(self.draw_line_pipe, self, "input_parameter")
            button_input.bind(on_press= buttoncallbackin)
            checkbox.was_active = True

        elif value == False and checkbox.was_active == True: #Destildo la opcion, elimino el boton
            for button in self.ids.inputs.children:
                if isinstance(button, MyParameterButton):
                    if button.button_id == 'input_parameter':
                        self.ids.inputs.remove_widget(button)
                        checkbox.was_active = False #Reinicio por si lo quiero volver a tildar en el
    
    def add_output_buttons(self, checkbox, value):
        if value and checkbox.was_active == False: # checkbox activado, No permito crear el mismo boton mas de una vez
            dir = str(Path(__file__).parent.absolute())
            if checkbox.parameter_text == 'dst':
                source = dir + '\\icons\\image_icon1.png'
            elif checkbox.parameter_text == 'retval':
                source = dir + '\\icons\\retval.png'
            button_output = MyParameterButton(button_id = 'output_parameter', parameter_text = checkbox.parameter_text, source = source,background_color = [0, 0, 0, 0])
            self.ids.outputs.size_hint = .2, (.2 + .05*(2))
            self.ids.outputs.add_widget(button_output)    
            button_output.bind(on_release = self.save_output)
            buttoncallbackin = partial(self.draw_line_pipe, self, "outputs")
            button_output.bind(on_press= buttoncallbackin)
            checkbox.was_active = True

        elif value == False and checkbox.was_active == True: #Destildo la opcion, elimino el boton
            for button in self.ids.outputs.children:
                if isinstance(button, MyParameterButton):
                    if button.button_id == 'output_parameter':
                        self.ids.outputs.remove_widget(button)
                        checkbox.was_active = False #Reinicio por si lo quiero volver a tildar en el
        
    def save_input(self,obj):
        print(obj.text)
        self.popup_save_input = Popup(title='Select %s' %obj.text, size_hint=(None, None), size= (500,500), background = "icons\\background_mainfloat.png", separator_color=(51/255,83/255,158/255,1), title_align="center")
        self.popup_save_input.open()
        pass

    def save_output(self,obj):
        print(obj.text)
        self.popup_save_output = Popup(title='Select %s' %obj.text, size_hint=(None, None), size= (500,500), background = "icons\\background_mainfloat.png", separator_color=(51/255,83/255,158/255,1), title_align="center")
        self.popup_save_output.open()
        pass

    def search_widgets(self, refresh=False):
        dir = str(Path(__file__).parent.absolute())
        with os.scandir(dir + '\\funciones') as json_files:
            for element in json_files:
                json_file = open(element)
                data = json.load(json_file)
                if refresh == True: #llamada de funcion restore
                    self.popup_bloque.ids.variable_box.clear_widgets()
                
                for key in data:
                    if key['nombre'] == self.funcion.nombre:
                        
                        for item in key['parameters_order']:
                            if item == 'dst':
                                self.parameters[item] = None
                            else:
                                self.parameters[item] = 'no input'

                            checkbox = MyCheckBox(item, False, size_hint=(1,1))                            
                            checkbox.bind(active=self.add_input_buttons)
                            self.popup_bloque.ids.opcional_checkbox_input.add_widget(checkbox)
                        
                        for item in key['outputs_order']:
                            self.outputs[item] = 'no output'
                            checkbox = MyCheckBox(item, False, size_hint=(1,1))
                            checkbox.bind(active=self.add_output_buttons)
                            self.popup_bloque.ids.opcional_checkbox_output.add_widget(checkbox)

                        #asignacion a variable del numero de inputs y outputs tipo imagen
                        self.in_images = key['input images']
                        self.out_images = key['output images']

                        for wid in key['widgets']:
                            if 'slider' in wid:
                                for item in wid['slider']:
                                    self.parameters[item['label']]=item['value'] #save parameter value in the order of the JSON file
                                    new_slider = MySlider(item['min'],item['max'],item['step'],item['value'],item['label'], size_hint= (.9, 1), pos_hint= {'center_x':0.5, 'center_y':1})
                                    self.popup_bloque.ids.variable_box.add_widget(new_slider)
                            if 'spinner' in wid:
                                for item in wid['spinner']:
                                    self.parameters[item['label']]=eval(item['text'])
                                    new_spinner = MySpinner(item['text'],item['values'],item['label'], size_hint= (.9, 1), pos_hint= {'center_x':0.5})
                                    self.popup_bloque.ids.variable_box.add_widget(new_spinner)
                            if 'image spinner' in wid:
                                for item in wid['image spinner']:
                                    self.parameters[item['label']]=eval(item['text'])
                                    new_image_spinner = MyImageSpinner(item['text'],item['label'], item['options'], size_hint= (.9, 1), pos_hint= {'center_x':0.5})
                                    self.popup_bloque.ids.variable_box.add_widget(new_image_spinner)
                            if 'especial' in wid:
                                eval(wid['especial'])
                            if 'toggle' in wid:
                                for item in wid['toggle']:
                                    self.parameters[item['label']]=item['text 2']
                                    new_toggle = MyToggleButton(item['text 1'], item['text 2'], item['label'], size_hint= (.9, 1), pos_hint= {'center_x':0.5})
                                    self.popup_bloque.ids.variable_box.add_widget(new_toggle)
                            if 'size' in wid:
                                for item in wid['size']:
                                    self.parameters[item['label']]=((item['value 1'],item['value 2']))
                                    if item['specification']:
                                        new_size_box = MySize(item['label'],item['input type 1'],item['input type 2'],item['value 1'], item['value 2'], item['specification']==True, size_hint= (.9, 1), pos_hint= {'center_x':0.5})
                                    self.popup_bloque.ids.variable_box.add_widget(new_size_box)
                            if 'text input' in wid:
                                for item in wid['text input']:
                                    self.parameters[item['label']]=item['text'] #save parameter value in the order of the JSON file
                                    new_text_input = MyTextInput(item['label'],item['input type'],item['text'], size_hint= (.9, 1), pos_hint= {'center_x':0.5})
                                    self.popup_bloque.ids.variable_box.add_widget(new_text_input)
                            if 'assign' in wid:
                                for item in wid['assign']:
                                    self.parameters[item['label']] = eval(item['text'])                        

    # para class Parameters_Popup
    def save_parameters(self):
        root = self.popup_bloque.ids.variable_box
        for child in root.children:
            if isinstance(child, MySlider):
                child.value = child.value_new
                self.parameters[child.slider_label] = child.value
            if isinstance(child, MySpinner):
                self.parameters[child.spinner_label] = eval(child.ids.spinner.text)
            if isinstance(child, MyImageSpinner):
                self.parameters[child.spinner_label] = eval(child.ids.spinner.text)
            if isinstance(child, MySize):
                self.parameters[child.text_label] = (int(child.ids.text_input_1.text), int(child.ids.text_input_2.text))
            if isinstance(child, MyTextInput):
                self.parameters[child.text_label]= child.value
            if isinstance(child, MyToggleButton):
                self.parameters[child.label] = child.down
        print("Parametros ", self.funcion.nombre, ":", self.parameters)   
        
        self.parent.parent.parent.run_pipes_until(self.scatter_id)
        #self.evaluate()
        #MODIFICAR
        #if self.funcion.nombre == "Threshold":
        #    cv2.imshow('Imagen evaluada',self.outputs[1])
        #else:
        #    cv2.imshow('Imagen evaluada', self.outputs)
        #cv2.waitKey(0)
        #self.try_evaluate(self.parameters)
        self.parameters_popup.dismiss()
        self.update_line(self)
        
    def cancel_parameters(self):
        root = self.popup_bloque.ids.variable_box                           
        for child in root.children:
            if isinstance(child, MySlider):
                child.ids.slider.value = child.value
            if isinstance(child, MySpinner):
                child.ids.spinner.text = child.text

        self.parameters_popup.dismiss()
        self.try_evaluate(self.parameters)
        self.update_line(self)
    
    #Para volver a valores iniciales del JSON
    def restore_parameters(self):
        self.search_widgets(refresh=True) #seria como un refresh a factory values
        self.parameters_popup.dismiss()  

    def try_parameters(self):
        root = self.popup_bloque.ids.variable_box
        parameters_aux = self.parameters.copy()
        for child in root.children:
            if isinstance(child, MySlider):
                parameters_aux[child.slider_label] = child.value_new
            if isinstance(child, MySpinner):
                parameters_aux[child.spinner_label] = eval(child.ids.spinner.text)
            if isinstance(child, MyImageSpinner):
                parameters_aux[child.spinner_label] = eval(child.ids.spinner.text)
            if isinstance(child, MySize):
                parameters_aux[child.text_label] = (int(child.ids.text_input_1.text), int(child.ids.text_input_2.text))
            if isinstance(child, MyTextInput):
                parameters_aux[child.text_label]= child.value
            if isinstance(child, MyToggleButton):
                parameters_aux[child.label] = child.down

        self.try_evaluate(parameters_aux)
        
    def imshow(self):
        self.popup_bloque.ids.boxlay.clear_widgets()
        button = Button(text='Close', size_hint= (1,0.09), halign = 'right', valign = 'bottom')
        self.popup_bloque.ids.boxlay.add_widget(button)
        if self.inputs != []:
            if isinstance(self.inputs, np.ndarray):
                image = self.inputs
            else:
                for element in self.inputs:
                    if isinstance(element, np.ndarray):
                        image = element
            
            # create a Texture the correct size and format for the image
            texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt= self.colorfmt)
            #texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt= 'luminance')
            #para mostrar imagen en blanco y negro: colorfmt='luminance'

            # copy the data into the texture
            texture.blit_buffer(image.tobytes(order=None), colorfmt= self.colorfmt, bufferfmt='ubyte')

            # flip the texture
            texture.flip_vertical()

            # actually put the texture in the kivy Image widget
            self.popup_bloque.add_widget(Image(color = (1,1,1,1), texture = texture))

    def duplicate(self):
        scatter_duplicate = self.parent.parent.parent.new_bloque(self.funcion.nombre)
        scatter_duplicate.parameters = self.parameters.copy()
        if self.funcion.nombre == "Load Image":
            scatter_duplicate.inputs = self.inputs.copy()
        else:
            scatter_duplicate.duplicate_parameters()
        self.parameters_popup.dismiss()

    def duplicate_parameters(self):
        root = self.popup_bloque.ids.variable_box
        for child in root.children:
            if isinstance(child, MySlider):
                child.value = self.parameters[child.slider_label]
            if isinstance(child, MySpinner):
                for value in child.ids.spinner.values:
                    if eval(value) == self.parameters[child.spinner_label]:
                        child.ids.spinner.text = value
                        break
            if isinstance(child, MyImageSpinner):
                child.ids.spinner.text = child.ids.spinner.values[self.parameters[child.spinner_label]]
            if isinstance(child, MySize):
                child.ids.text_input_1.text = str(self.parameters[child.text_label][0])
                child.ids.text_input_2.text = str(self.parameters[child.text_label][1])
            if isinstance(child, MyTextInput):
                child.value = str(self.parameters[child.text_label]) #chequear
            if isinstance(child, MyToggleButton):
                child.down = self.parameters[child.label] #chequear

    # para class LoadDialog
    def load_dialog(self):
        self.popup_bloque.ids.parameters_popup.clear_widgets()
        self.popup_bloque.ids.parameters_popup.add_widget(LoadDialog(load=self.load, cancel=self.dismiss_popup))

    def load(self, path, filename):
        filename = os.path.join(path, filename[0])
        
        if self.inputs == []:
            self.inputs.append(filename)
        else:
            self.inputs[0]=filename
            self.parent.parent.parent.run_pipes_until(self.scatter_id)
        self.dismiss_popup()
    
    def dismiss_popup(self):
        self.parameters_popup.dismiss()

class MyScatterLayout(Bloque, ScatterLayout):

    def __init__(self, draw_line_pipe, update_line, delete_scatter, funcion, scatter_id, **kwargs):
        #inicializo scatter layout
        ScatterLayout.__init__(self, **kwargs)
        #inicializo bloque
        super().__init__(funcion)
        #add content of parameters for the function to the popup
        #self.search_widgets()
        self.scatter_id = scatter_id
        self.parameters_popup = Popup(title="Modify Parameters", content=self.popup_bloque, size_hint=(None, None), size=(600, 500), auto_dismiss= False, background = "icons\\background_mainfloat.png", separator_color=(51/255,83/255,158/255,1), title_align="center")
        self.ids.funcion_scatter.text = self.funcion.nombre
        self.ids.funcion_scatter.font_size = 10
        self.draw_line_pipe = draw_line_pipe
        self.update_line = update_line
        self.delete_scatter = delete_scatter
        self.join_buttons()
        self.pos_min = 200, 200
        self.pos_max = 400, 200

    isShownMenu = kprop.BooleanProperty(True)
    move_lock = False
    scale_lock_left = False
    scale_lock_right = False
    scale_lock_top = False
    scale_lock_bottom = False

    def join_buttons(self):
        dir = str(Path(__file__).parent.absolute())
        with os.scandir(dir + '\\funciones') as json_files:
            for element in json_files:
                json_file = open(element)
                data = json.load(json_file)
                for key in data:
                    if key['nombre'] == self.funcion.nombre:
                        
                        if key['input images'] != 0:
                            cant_buttons_in = key ['input images']
                            
                            for n in range(cant_buttons_in):
                                button_input = MyInputButton(background_color = [0, 0, 0, 0])
                                button_input.source = "icons\image_icon1.png"
                                self.ids.inputs.size_hint = .2, (.2 + .05*cant_buttons_in)
                                self.ids.inputs.add_widget(button_input)
                                buttoncallbackin = partial(self.draw_line_pipe, self, "inputs")
                                button_input.bind(on_press= buttoncallbackin)
                        
                        if key['output images'] != 0:
                            cant_buttons_out = key ['output images']
                            cant = 0
                            
                            #for n in range(cant_buttons_out):
                            for item in key["outputs_order"]:
                                dir = str(Path(__file__).parent.absolute())
                                if item == 'dst':
                                    #source = r'C:\Users\trini\OneDrive\Favaloro\Tesis\Código\09-Septiembre\01_09_21\icons\image_icon1.png'
                                    source = dir + '\\icons\\image_icon1.png'
                                    button_output = MyParameterButton(button_id = 'outputs', parameter_text = item, source=source,background_color = [0, 0, 0, 0]) #genero los botones para ponerle los ids pero solo agrego los que son outp imagen
                                    if cant < cant_buttons_out:
                                        self.ids.outputs.size_hint = .2, (.2 + .05*cant_buttons_out)
                                        self.ids.outputs.add_widget(button_output)
                                        buttoncallbackout = partial(self.draw_line_pipe, self, "outputs")
                                        button_output.bind(on_press= buttoncallbackout)
                                        cant = cant + 1
                                #elif item == 'retval':
                                    #source = r'C:\Users\trini\OneDrive\Favaloro\Tesis\Código\09-Septiembre\01_09_21\icons\retval.png'
                                
    def on_touch_up(self, touch):
        self.move_lock = False
        self.scale_lock_left = False
        self.scale_lock_right = False
        self.scale_lock_top = False
        self.scale_lock_bottom = False
        if touch.grab_current is self:
            touch.ungrab(self)
            x = self.pos[0] / 10
            x = round(x, 0)
            x = x * 10
            y = self.pos[1] / 10
            y = round(y, 0)
            y = y * 10
            self.pos = x, y
            return super(MyScatterLayout, self).on_touch_up(touch)

    def transform_with_touch(self, touch):
        try:
            changed = False
            x = self.bbox[0][0]
            y = self.bbox[0][1]
            width = self.bbox[1][0]
            height = self.bbox[1][1]
            mid_x = x + width / 2
            mid_y = y + height / 2
            inner_width = width * 0.5
            inner_height = height * 0.5
            left = mid_x - (inner_width / 2)
            right = mid_x + (inner_width / 2)
            top = mid_y + (inner_height / 2)
            bottom = mid_y - (inner_height / 2)

                # just do a simple one finger drag
            if len(self._touches) == self.translation_touches:
                # _last_touch_pos has last pos in correct parent space,
                # just like incoming touch
                dx = (touch.x - self._last_touch_pos[touch][0]) \
                    * self.do_translation_x
                dy = (touch.y - self._last_touch_pos[touch][1]) \
                    * self.do_translation_y
                dx = dx / self.translation_touches
                dy = dy / self.translation_touches
                if (touch.x > left and touch.x < right and touch.y < top and touch.y > bottom or self.move_lock) and not self.scale_lock_left and not self.scale_lock_right and not self.scale_lock_top and not self.scale_lock_bottom:
                    self.move_lock = True
                    self.apply_transform(ktransf.Matrix().translate(dx, dy, 0))
                    changed = True

            change_x = touch.x - self.prev_x
            change_y = touch.y - self.prev_y
            anchor_sign = 1
            sign = 1
            if abs(change_x) >= 9 and not self.move_lock and not self.scale_lock_top and not self.scale_lock_bottom:
                if change_x < 0:
                    sign = -1
                if (touch.x < left or self.scale_lock_left) and not self.scale_lock_right:
                    self.scale_lock_left = True
                    self.pos = (self.pos[0] + (sign * 10), self.pos[1])
                    anchor_sign = -1
                elif (touch.x > right or self.scale_lock_right) and not self.scale_lock_left:
                    self.scale_lock_right = True

                
                #self.size[0] = self.size[0] + (sign * anchor_sign * 10)
                #Cambie la linea de arriba por la de abajo para que no de error
                setattr(self.size, 0, self.size[0] + (sign * anchor_sign * 10))
                self.prev_x = touch.x
                changed = True
            if abs(change_y) >= 9 and not self.move_lock and not self.scale_lock_left and not self.scale_lock_right:
                if change_y < 0:
                    sign = -1
                if (touch.y > top or self.scale_lock_top) and not self.scale_lock_bottom:
                    self.scale_lock_top = True
                elif (touch.y < bottom or self.scale_lock_bottom) and not self.scale_lock_top:
                    self.scale_lock_bottom = True
                    self.pos = (self.pos[0], self.pos[1] + (sign * 10))
                    anchor_sign = -1
                #self.size[1] = self.size[1] + (sign * anchor_sign * 10)
                #Cambie la linea de arriba por la de abajo para que no de error
                setattr(self.size, 1, self.size[1] + (sign * anchor_sign * 10))
                self.prev_y = touch.y
                changed = True
            return changed
        except TypeError:
            pass

    def on_touch_down(self, touch):
        
        if touch.is_double_tap:
            if self.inputs != []:
                if self.funcion.nombre != "Load Image":
                    if self.outputs != {}:
                        self.view_popup_image(self.outputs.values())
            self.parameters_popup.open()  
        #elif touch.button == 'right':
        #    self.ids.context_menu.show(touch.pos[0], touch.pos[1])
        else:       
            x, y = touch.x, touch.y
            self.prev_x = touch.x
            self.prev_y = touch.y
            # if the touch isnt on the widget we do nothing
            if not self.do_collide_after_children:
                if not self.collide_point(x, y):
                    return False
            
            # let the child widgets handle the event if they want
            touch.push()

            #touch.apply_transform_2d(self.to_local)
            #if super(Scatter,self).on_touch_down(touch):

            #Cambie las dos lineas de arriba por esto para que no de error:
            if super().on_touch_down(touch):
                # ensure children don't have to do it themselves
                if 'multitouch_sim' in touch.profile:
                    touch.multitouch_sim = True
                touch.pop()
                self._bring_to_front(touch)
                return True
            # whatever the result, don't forget to pop your transformation
            # after the call, so the coordinate will be back in parent space
            touch.pop()

            # if our child didn't do anything, and if we don't have any active
            # interaction control, then don't accept the touch.
            if not self.do_translation_x and \
                    not self.do_translation_y and \
                    not self.do_rotation and \
                    not self.do_scale:
                return False

            if self.do_collide_after_children:
                if not self.collide_point(x, y):
                    return False

            if 'multitouch_sim' in touch.profile:
                touch.multitouch_sim = True
            # grab the touch so we get all it later move events for sure
            self._bring_to_front(touch)
            touch.grab(self)
            self._touches.append(touch)
            self._last_touch_pos[touch] = touch.pos      
            
        return True
    
    def change_size(self):
        if self.ids.button_size.text == "-":
            self.size = (200, 70)
            self.ids.button_size.text = "+"
        elif self.ids.button_size.text == "+":
            self.size = (200, 100)
            self.ids.button_size.text = "-"

    def view_scatter_image(self, outputs):
        #Muestro imagen en Popup
        if isinstance(outputs, np.ndarray):
            image = outputs
        else:
            for element in outputs:
                if isinstance(element, np.ndarray):
                    image = element
        
        #cv2.imshow('Imagen resultado del pipe', image)
        #cv2.waitKey(0)

        # create a Texture the correct size and format for the image
        texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt= self.colorfmt)
        #texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt= 'luminance')
        #para mostrar imagen en blanco y negro: colorfmt='luminance'

        # copy the data into the texture
        texture.blit_buffer(image.tobytes(order=None), colorfmt= self.colorfmt, bufferfmt='ubyte')

        # flip the texture
        texture.flip_vertical()

        self.ids.image_scatter.color = (1,1,1,1)
        self.ids.image_scatter.texture = texture
        #return self.ids.image_scatter

class MySize(GridLayout):
    text_label = kprop.StringProperty()
    input_type_1 = kprop.StringProperty()
    input_type_2 = kprop.StringProperty()
    value_1 = kprop.NumericProperty() 
    value_2 = kprop.NumericProperty() 

    def __init__(self, label, input_type_1, input_type_2, value_1, value_2, odd = False, even = True, **kwargs):
        super(MySize, self).__init__(**kwargs)
        self.text_label = label
        self.input_type_1 = input_type_1 #None, ‘int’ (string), or ‘float’ (string), or a callable
        self.input_type_2 = input_type_2 #None, ‘int’ (string), or ‘float’ (string), or a callable
        self.value_1 = value_1
        self.value_2 = value_2
        
        if odd == True: self.odd = True
        if even == True: self.even = True

        self.cols = 6
        self.row_force_default=True
        self.row_default_height=30
        #self.size_hint=(1, 0.05)

class MyToggleButton(GridLayout):
    text_1 = kprop.StringProperty()
    text_2 = kprop.StringProperty()
    label = kprop.StringProperty()

    def __init__(self, text_1, text_2, label, **kwargs):
        super(MyToggleButton, self).__init__(**kwargs)
        self.text_1 = text_1
        self.text_2 = text_2
        self.down = text_2
        self.label = label
        self.cols = 3
        self.row_force_default=True
        self.row_default_height=40
        #self.size_hint=(1, 0.05)
    
    def state_change(self, text):
        self.down = text

class MyTextInput(GridLayout):
    text_label = kprop.StringProperty()
    input_type = kprop.StringProperty()
    value = kprop.NumericProperty() 

    def __init__(self, label, input_type, value, **kwargs):
        super(MyTextInput, self).__init__(**kwargs)
        self.text_label = label
        self.input_type = input_type #None, ‘int’ (string), or ‘float’ (string), or a callable
        self.value = value
        self.cols = 2
        self.row_force_default=True
        self.row_default_height=40
        #self.size_hint=(1, 0.05)

class MyButton(Button):
    def on_touch_down(self, touch):
        return False

class MyBubble(Bubble):
    pass

class MySlider(GridLayout):
    minimum = kprop.NumericProperty() #defaults to 0
    maximum = kprop.NumericProperty()
    step = kprop.NumericProperty() #defaults to 1
    value = kprop.NumericProperty() #defaults to 0
    slider_label = kprop.StringProperty()
    value_new = kprop.NumericProperty()
    slider_value = kprop.NumericProperty()

    def __init__(self, minimum, maximum, step, value, slider_label, **kwargs):
        super(MySlider, self).__init__(**kwargs)
        self.minimum = minimum #defaults to 0
        self.maximum = maximum
        self.step = step #defaults to 1
        self.value = value #defaults to 0
        self.slider_label = slider_label
        self.cols = 2
        self.row_force_default=True
        self.row_default_height=40        
    
    def slider_change(self, value):
        self.value_new = value

class MySpinner(GridLayout):
    text = kprop.StringProperty() #default value shown
    values = kprop.ListProperty() #available values in dropdown list
    spinner_label = kprop.StringProperty()
    codes = kprop.NumericProperty()

    def __init__(self, text, values, spinner_label, **kwargs):
        super(MySpinner, self).__init__(**kwargs)
        self.text = text
        self.values = values
        self.spinner_label = spinner_label
        self.cols = 2
        self.rows =2
        self.row_force_default=True
        self.row_default_height=40 

    def get_code(self, text_in):
        try:
            if text_in in self.values:
                self.text = text_in
            else: #si escribe un codigo no valido, se pone el primero de la lista por default
                self.text = self.values[0]
                raise SyntaxError

        except SyntaxError: 
            print("SyntaxError: Escribir un codigo valido")

class MyImageSpinner(GridLayout):
    options = kprop.DictProperty()
    text = kprop.StringProperty() #default value shown

    def __init__(self, text, spinner_label, options, **kwargs):
        super(MyImageSpinner, self).__init__(**kwargs)
        self.text = text
        self.spinner_label = spinner_label
        self.options = options
        
        self.rows = 2
        self.row_force_default=True
        self.row_default_height=40 

        self.ids.spinner.values = self.options.keys()
        self.ids.imageplan.source = self.options.get(self.ids.spinner.text)

class MyCheckBox(ContextMenuItem, GridLayout):
    parameter_text = kprop.StringProperty()
    active = kprop.BooleanProperty()

    def __init__(self, parameter_text, active, **kwargs):
        super(MyCheckBox, self).__init__(**kwargs)
        self.parameter_text = parameter_text
        self.active = active
        self.was_active = False
    
    def on_checkbox_active(self,value):
        self.active = value
            
class MyParameterButton(Button):
    def __init__(self, button_id, parameter_text, source, **kwargs):
        super(MyParameterButton, self).__init__(**kwargs)
        self.button_id = button_id #para identificar que es de tipo parametro con id='input parameter' o 'output parameter'
        self.parameter_text = parameter_text #para guardar el texto con el que se identifica el parametro en el diccionario (parameters o outputs)
        self.source = source
        self.ids.param_imag.source = source
        #Window.bind(mouse_pos=self.on_mouse_pos)# binding[subscribe]Event handling method of mouse position change
    
    #def on_touch_down(self, touch):
    #    if touch.is_double_tap:  
    
class MyIconButton(Button):
    source = kprop.StringProperty()
    pass

class MyIconButtonLeft(Button):
    source = kprop.StringProperty()
    pass

class MyInputButton(Button):
    source = kprop.StringProperty()
    pass

class Parameters_Popup(FloatLayout):
    save_parameters = kprop.ObjectProperty()
    cancel_parameters = kprop.ObjectProperty(None)
    restore_parameters = kprop.ObjectProperty(None)
    try_parameters = kprop.ObjectProperty(None)
    duplicate = kprop.ObjectProperty(None)
    pass

class LoadDialog(GridLayout):
    load = kprop.ObjectProperty()
    cancel = kprop.ObjectProperty(None)

    def selected(self, filename):
        try:
            self.ids.load_image.source = filename[0]
            self.ids.load_image.color = (1,1,1,1)
        except:
            pass
    
    def new_path(self, newpath):
        try:
            self.ids.filechooser.path = newpath
            
        except:
            print(newpath)
            pass

class RoundedButton(Button):
    line_color = kprop.ColorProperty()
    pass
    #def __init__(self, line_color, **kwargs):
     #   super(RoundedButton, self).__init__(**kwargs)
      #  self.line_color = line_color
    
