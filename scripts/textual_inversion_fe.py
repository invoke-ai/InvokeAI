#!/usr/bin/env python

import npyscreen
import os
import sys
import curses
import re
from ldm.invoke.globals import Globals
from omegaconf import OmegaConf
from pathlib import Path
from typing import List
import argparse

class textualInversionForm(npyscreen.FormMultiPageAction):
    resolutions = [512, 768, 1024]
    lr_schedulers = [
                "linear", "cosine", "cosine_with_restarts",
                "polynomial","constant", "constant_with_warmup"
    ]

    def afterEditing(self):
        self.parentApp.setNextForm(None)

    def create(self):
        self.model_names, default = self.get_model_names()
        default_initializer_token = '★'
        default_placeholder_token = ''

        self.model = self.add_widget_intelligent(
            npyscreen.TitleSelectOne,
            name='Model Name',
            values=self.model_names,
            value=default,
            max_height=len(self.model_names)+1
        )
        self.placeholder_token = self.add_widget_intelligent(
            npyscreen.TitleText,
            name="Placeholder token",
            value='',
        )
        self.nextrely -= 1
        self.nextrelx += 30
        self.prompt_token = self.add_widget_intelligent(
            npyscreen.FixedText,
            name="Trigger term for use in prompt",
            value='',
        )
        self.nextrelx -= 30
        self.initializer_token = self.add_widget_intelligent(
            npyscreen.TitleText,
            name="Initializer token",
            value=default_initializer_token,
        )
        self.learnable_property = self.add_widget_intelligent(
            npyscreen.TitleSelectOne,
            name="Learnable property",
            values=['object','style'],
            value=0,
            max_height=3,
        )
        self.train_data_dir = self.add_widget_intelligent(
            npyscreen.TitleFilenameCombo,
            name='Data Training Directory',
            select_dir=True,
            must_exist=True,
            value=Path(Globals.root) / 'training-data' / default_placeholder_token
        )
        self.output_dir = self.add_widget_intelligent(
            npyscreen.TitleFilenameCombo,
            name='Output Destination Directory',
            select_dir=True,
            must_exist=False,
            value=Path(Globals.root) / 'trained-checkpoints' / default_placeholder_token
        )
        self.resolution = self.add_widget_intelligent(
            npyscreen.TitleSelectOne,
            name='Image resolution (pixels)',
            values = self.resolutions,
            value=0,
            scroll_exit = True,
            max_height=4,
        )
        self.center_crop = self.add_widget_intelligent(
            npyscreen.Checkbox,
            name="Center crop images before resizing to resolution",
            value=False,
        )
        self.train_batch_size = self.add_widget_intelligent(
            npyscreen.TitleSlider,
            name='Batch Size',
            out_of=50,
            step=1,
            lowest=1,
            value=16
        )
        self.gradient_accumulation_steps = self.add_widget_intelligent(
            npyscreen.TitleSlider,
            name='Gradient Accumulation Steps',
            out_of=10,
            step=1,
            lowest=1,
            value=4
        )
        self.max_train_steps = self.add_widget_intelligent(
            npyscreen.TitleSlider,
            name='Max Training Steps',
            out_of=10000,
            step=500,
            lowest=1,
            value=3000
        )
        self.learning_rate = self.add_widget_intelligent(
            npyscreen.TitleText,
            name="Learning Rate",
            value='5.0e-04',
        )
        self.scale_lr = self.add_widget_intelligent(
            npyscreen.Checkbox,
            name="Scale learning rate by number GPUs, steps and batch size",
            value=True
        )
        self.enable_xformers_memory_efficient_attention = self.add_widget_intelligent(
            npyscreen.Checkbox,
            name="Use xformers acceleration",
            value=False,
        )
        self.lr_scheduler = self.add_widget_intelligent(
            npyscreen.TitleSelectOne,
            name='Learning rate scheduler',
            values = self.lr_schedulers,
            max_height=7,
            scroll_exit = True,
            value=4)
        self.lr_warmup_steps = self.add_widget_intelligent(
            npyscreen.TitleSlider,
            name='Warmup Steps',
            out_of=100,
            step=1,
            lowest=0,
            value=0
        )
        self.placeholder_token.when_value_edited = self.initializer_changed

    def initializer_changed(self):
        placeholder = self.placeholder_token.value
        self.prompt_token.value = f'(Trigger by using <{placeholder}> in your prompts)'
        self.train_data_dir.value = Path(Globals.root) / 'training-data' / placeholder
        self.output_dir.value = Path(Globals.root) / 'embeddings' / placeholder
        
    def on_ok(self):
        if self.validate_field_values():
            self.parentApp.setNextForm(None)
            self.editing = False
            self.parentApp.ti_arguments = self.marshall_arguments()
            npyscreen.notify('Launching textual inversion training. This will take a while...')
            # The module load takes a while, so we do it while the form and message are still up
            import ldm.invoke.textual_inversion_training
        else:
            self.editing = True

    def validate_field_values(self)->bool:
        bad_fields = []
        if self.model.value is None:
            bad_fields.append('Model Name must correspond to a known model in models.yaml')
        if self.train_data_dir.value is None:
            bad_fields.append('Data Training Directory cannot be empty')
        if self.output_dir.value is None:
            bad_fields.append('The Output Destination Directory cannot be empty')
        if len(bad_fields) > 0:
            message = 'The following problems were detected and must be corrected:'
            for problem in bad_fields:
                message += f'\n* {problem}'
            npyscreen.notify_confirm(message)
            return False
        else:
            return True

    def get_model_names(self)->(List[str],int):
        conf = OmegaConf.load(os.path.join(Globals.root,'configs/models.yaml'))
        model_names = list(conf.keys())
        defaults = [idx for idx in range(len(model_names)) if 'default' in conf[model_names[idx]]]
        return (model_names,defaults[0])

    def marshall_arguments(self):
        args = dict()

        # the choices
        args.update(
            model = self.model_names[self.model.value[0]],
            resolution = self.resolutions[self.resolution.value[0]],
            lr_scheduler = self.lr_schedulers[self.lr_scheduler.value[0]]
        )

        # all the strings
        for attr in ('initializer_token','placeholder_token','train_data_dir','output_dir','scale_lr'):
            args[attr] = getattr(self,attr).value
        # all the integers
        for attr in ('train_batch_size','gradient_accumulation_steps',
                     'max_train_steps','lr_warmup_steps'):
            args[attr] = int(getattr(self,attr).value)

        # the floats (just one)
        args.update(
            learning_rate = float(self.learning_rate.value)
        )
        return args

class MyApplication(npyscreen.NPSAppManaged):
    def __init__(self):
        super().__init__()
        self.ti_arguments=None

    def onStart(self):
        npyscreen.setTheme(npyscreen.Themes.DefaultTheme)
        self.main = self.addForm('MAIN', textualInversionForm, name='Textual Inversion Settings')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='InvokeAI textual inversion training')
    parser.add_argument(
        '--root_dir','--root-dir',
        type=Path,
        default=Globals.root,
        help='Path to the invokeai runtime directory',
    )
    args = parser.parse_args()
    Globals.root = args.root_dir
    
    myapplication = MyApplication()
    myapplication.run()
    
    from ldm.invoke.textual_inversion_training import do_textual_inversion_training
    if args := myapplication.ti_arguments:
        os.makedirs(args['output_dir'],exist_ok=True)
        do_textual_inversion_training(**args)
    
