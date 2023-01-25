#!/usr/bin/env python

import npyscreen
import os
import sys
import traceback
import argparse
from ldm.invoke.globals import Globals, global_set_root, global_cache_dir, global_config_file
from ldm.invoke.model_manager import ModelManager
from omegaconf import OmegaConf
from pathlib import Path
from typing import List

class FloatSlider(npyscreen.Slider):
    # this is supposed to adjust display precision, but doesn't
    def translate_value(self):
        stri = "%3.2f / %3.2f" %(self.value, self.out_of)
        l = (len(str(self.out_of)))*2+4
        stri = stri.rjust(l)
        return stri

class FloatTitleSlider(npyscreen.TitleText):
    _entry_type = FloatSlider

class mergeModelsForm(npyscreen.FormMultiPageAction):

    interpolations = ['weighted_sum',
                      'sigmoid',
                      'inv_sigmoid',
                      'add_difference']

    def __init__(self, parentApp, name):
        self.parentApp = parentApp
        super().__init__(parentApp, name)

    @property
    def model_manager(self):
        return self.parentApp.model_manager

    def afterEditing(self):
        self.parentApp.setNextForm(None)

    def create(self):
        self.model_names = self.get_model_names()
        
        self.add_widget_intelligent(
            npyscreen.FixedText,
            name="Select up to three models to merge",
            value=''
        )
        self.model1 = self.add_widget_intelligent(
            npyscreen.TitleSelectOne,
            name='First Model:',
            values=self.model_names,
            value=0,
            max_height=len(self.model_names)+1
        )
        self.model2 = self.add_widget_intelligent(
            npyscreen.TitleSelectOne,
            name='Second Model:',
            values=self.model_names,
            value=1,
            max_height=len(self.model_names)+1
        )
        models_plus_none = self.model_names.copy()
        models_plus_none.insert(0,'None')
        self.model3 = self.add_widget_intelligent(
            npyscreen.TitleSelectOne,
            name='Third Model:',
            values=models_plus_none,
            value=0,
            max_height=len(self.model_names)+1,
        )
        
        for m in [self.model1,self.model2,self.model3]:
            m.when_value_edited = self.models_changed
            
        self.merge_method = self.add_widget_intelligent(
            npyscreen.TitleSelectOne,
            name='Merge Method:',
            values=self.interpolations,
            value=0,
            max_height=len(self.interpolations),
        )
        self.alpha = self.add_widget_intelligent(
            FloatTitleSlider,
            name='Weight (alpha) to assign to second and third models:',
            out_of=1,
            step=0.05,
            lowest=0,
            value=0.5,
        )
        self.force = self.add_widget_intelligent(
            npyscreen.Checkbox,
            name='Force merge of incompatible models',
            value=False,
        )
        self.merged_model_name = self.add_widget_intelligent(
            npyscreen.TitleText,
            name='Name for merged model',
            value='',
        )

    def models_changed(self):
        models = self.model1.values
        selected_model1 = self.model1.value[0]
        selected_model2 = self.model2.value[0]
        selected_model3 = self.model3.value[0]
        merged_model_name = f'{models[selected_model1]}+{models[selected_model2]}'
        self.merged_model_name.value = merged_model_name
        
        if selected_model3 > 0:
            self.merge_method.values=['add_difference'],
            self.merged_model_name.value += f'+{models[selected_model3]}'
        else:
            self.merge_method.values=self.interpolations
        self.merge_method.value=0
        
    def on_ok(self):
        if self.validate_field_values() and self.check_for_overwrite():
            self.parentApp.setNextForm(None)
            self.editing = False
            self.parentApp.merge_arguments = self.marshall_arguments()
            npyscreen.notify('Starting the merge...')
            import ldm.invoke.merge_diffusers  # this keeps the message up while diffusers loads
        else:
            self.editing = True

    def on_cancel(self):
        sys.exit(0)

    def marshall_arguments(self)->dict:
        model_names = self.model_names
        models = [
            model_names[self.model1.value[0]],
            model_names[self.model2.value[0]],
            ]
        if self.model3.value[0] > 0:
            models.append(model_names[self.model3.value[0]-1])

        args = dict(
            models=models,
            alpha = self.alpha.value,
            interp = self.interpolations[self.merge_method.value[0]],
            force = self.force.value,
            merged_model_name = self.merged_model_name.value,
        )
        return args

    def check_for_overwrite(self)->bool:
        model_out = self.merged_model_name.value
        if model_out not in self.model_names:
            return True
        else:
            return npyscreen.notify_yes_no(f'The chosen merged model destination, {model_out}, is already in use. Overwrite?')

    def validate_field_values(self)->bool:
        bad_fields = []
        model_names = self.model_names
        selected_models = set((model_names[self.model1.value[0]],model_names[self.model2.value[0]]))
        if self.model3.value[0] > 0:
            selected_models.add(model_names[self.model3.value[0]-1])
        if len(selected_models) < 2:
            bad_fields.append(f'Please select two or three DIFFERENT models to compare. You selected {selected_models}')
        if len(bad_fields) > 0:
            message = 'The following problems were detected and must be corrected:'
            for problem in bad_fields:
                message += f'\n* {problem}'
            npyscreen.notify_confirm(message)
            return False
        else:
            return True

    def get_model_names(self)->List[str]:
        model_names = [name for name in self.model_manager.model_names() if self.model_manager.model_info(name).get('format') == 'diffusers']
        print(model_names)
        return sorted(model_names)

class Mergeapp(npyscreen.NPSAppManaged):
    def __init__(self):
        super().__init__()
        conf = OmegaConf.load(global_config_file())
        self.model_manager = ModelManager(conf,'cpu','float16') # precision doesn't really matter here

    def onStart(self):
        npyscreen.setTheme(npyscreen.Themes.DefaultTheme)
        self.main = self.addForm('MAIN', mergeModelsForm, name='Merge Models Settings')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='InvokeAI textual inversion training')
    parser.add_argument(
        '--root_dir','--root-dir',
        type=Path,
        default=Globals.root,
        help='Path to the invokeai runtime directory',
    )
    args = parser.parse_args()
    global_set_root(args.root_dir)

    cache_dir = str(global_cache_dir('diffusers')) # because not clear the merge pipeline is honoring cache_dir
    os.environ['HF_HOME'] = cache_dir

    mergeapp = Mergeapp()
    mergeapp.run()

    args = mergeapp.merge_arguments
    args.update(cache_dir = cache_dir)
    from ldm.invoke.merge_diffusers import merge_diffusion_models

    try:
        merge_diffusion_models(**args)
        print(f'>> Models merged into new model: "{args["merged_model_name"]}".')
    except Exception as e:
        print(f'** An error occurred while merging the pipelines: {str(e)}')
        print('** DETAILS:')
        print(traceback.format_exc())
        sys.exit(-1)
