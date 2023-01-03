#!/usr/bin/env python

import npyscreen
import os
import sys
import curses
from ldm.invoke.globals import Globals
from omegaconf import OmegaConf
from pathlib import Path
from typing import List
from argparse import Namespace

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
        default_token = 'cat-toy'

        self.model = self.add_widget_intelligent(
            npyscreen.TitleSelectOne,
            name='Model Name',
            values=self.model_names,
            value=default,
            max_height=len(self.model_names)+1
        )
        self.initializer_token = self.add_widget_intelligent(
            npyscreen.TitleText,
            name="Initializer Token",
            value=default_token,
        )
        self.placeholder_token = self.add_widget_intelligent(
            npyscreen.TitleText,
            name="Placeholder Token",
            value=f'<{default_token}>'
        )
        self.train_data_dir = self.add_widget_intelligent(
            npyscreen.TitleFilenameCombo,
            name='Data Training Directory',
            select_dir=True,
            must_exist=True,
            value=Path(Globals.root) / 'training-data' / default_token
        )
        self.output_dir = self.add_widget_intelligent(
            npyscreen.TitleFilenameCombo,
            name='Output Destination Directory',
            select_dir=True,
            must_exist=False,
            value=Path(Globals.root) / 'embeddings' / default_token
        )
        self.resolution = self.add_widget_intelligent(
            npyscreen.TitleSelectOne,
            name='Resolution',
            values = self.resolutions,
            value=0,
            scroll_exit = True,
            max_height=3
        )
        self.train_batch_size = self.add_widget_intelligent(
            npyscreen.TitleSlider,
            name='Batch Size',
            out_of=10,
            step=1,
            lowest=1,
            value=1
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
        self.initializer_token.when_value_edited = self.initializer_changed

    def initializer_changed(self):
        new_value = self.initializer_token.value
        self.placeholder_token.value = f'<{new_value}>'
        self.train_data_dir.value = Path(Globals.root) / 'training-data' / new_value
        self.output_dir.value = Path(Globals.root) / 'embeddings' / new_value
        
    def on_ok(self):
        if self.validate_field_values():
            self.parentApp.setNextForm(None)
            self.editing = False
            self.run_textual_inversion()
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

    def create_namespace(self):
        args = Namespace()

        # the choices
        args.model = self.model_names[self.model.value[0]]
        args.resolution = self.resolutions[self.resolution.value[0]]
        args.lr_scheduler = self.lr_schedulers[self.lr_scheduler.value[0]]

        # all the strings
        for attr in ('initializer_token','placeholder_token','train_data_dir','output_dir','scale_lr'):
            setattr(args,attr,getattr(self,attr).value)
        # all the integers
        for attr in ('train_batch_size','gradient_accumulation_steps',
                     'max_train_steps','lr_warmup_steps'):
            setattr(args,attr,int(getattr(self,attr).value))
        # the floats (just one)
        args.learning_rate = float(self.learning_rate.value)
        return args

    def run_textual_inversion(self):
        npyscreen.notify('Launching textual inversion training. This will take a while...')
        from ldm.invoke.textual_inversion_training import do_textual_inversion_training, parse_args
        args = parse_args()
        args.root_dir = Globals.root
        do_textual_inversion_training(args)

class MyApplication(npyscreen.NPSAppManaged):
    def onStart(self):
        npyscreen.setTheme(npyscreen.Themes.DefaultTheme)
        self.main = self.addForm('MAIN', textualInversionForm, name='Textual Inversion Settings')

if __name__ == '__main__':
   TestApp = MyApplication().run()
