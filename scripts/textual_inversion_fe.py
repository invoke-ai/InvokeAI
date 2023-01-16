#!/usr/bin/env python

import npyscreen
import os
import sys
import re
import shutil
import traceback
from ldm.invoke.globals import Globals, global_set_root
from omegaconf import OmegaConf
from pathlib import Path
from typing import List
import argparse

TRAINING_DATA = 'training-data'
TRAINING_DIR = 'text-inversion-training'
CONF_FILE = 'preferences.conf'

class textualInversionForm(npyscreen.FormMultiPageAction):
    resolutions = [512, 768, 1024]
    lr_schedulers = [
                "linear", "cosine", "cosine_with_restarts",
                "polynomial","constant", "constant_with_warmup"
    ]
    precisions = ['no','fp16','bf16']
    learnable_properties = ['object','style']

    def __init__(self, parentApp, name, saved_args=None):
        self.saved_args = saved_args or {}
        super().__init__(parentApp, name)

    def afterEditing(self):
        self.parentApp.setNextForm(None)

    def create(self):
        self.model_names, default = self.get_model_names()
        default_initializer_token = '★'
        default_placeholder_token = ''
        saved_args = self.saved_args

        try:
            default = self.model_names.index(saved_args['model'])
        except:
            pass

        self.model = self.add_widget_intelligent(
            npyscreen.TitleSelectOne,
            name='Model Name:',
            values=self.model_names,
            value=default,
            max_height=len(self.model_names)+1
        )
        self.placeholder_token = self.add_widget_intelligent(
            npyscreen.TitleText,
            name='Trigger Term:',
            value='', # saved_args.get('placeholder_token',''), # to restore previous term
        )
        self.placeholder_token.when_value_edited = self.initializer_changed
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
            name='Initializer:',
            value=saved_args.get('initializer_token',default_initializer_token),
        )
        self.resume_from_checkpoint = self.add_widget_intelligent(
            npyscreen.Checkbox,
            name="Resume from last saved checkpoint",
            value=False,
        )
        self.learnable_property = self.add_widget_intelligent(
            npyscreen.TitleSelectOne,
            name="Learnable property:",
            values=self.learnable_properties,
            value=self.learnable_properties.index(saved_args.get('learnable_property','object')),
            max_height=4,
        )
        self.train_data_dir = self.add_widget_intelligent(
            npyscreen.TitleFilenameCombo,
            name='Data Training Directory:',
            select_dir=True,
            must_exist=True,
            value=saved_args.get('train_data_dir',Path(Globals.root) / TRAINING_DATA / default_placeholder_token)
        )
        self.output_dir = self.add_widget_intelligent(
            npyscreen.TitleFilenameCombo,
            name='Output Destination Directory:',
            select_dir=True,
            must_exist=False,
            value=saved_args.get('output_dir',Path(Globals.root) / TRAINING_DIR / default_placeholder_token)
        )
        self.resolution = self.add_widget_intelligent(
            npyscreen.TitleSelectOne,
            name='Image resolution (pixels):',
            values = self.resolutions,
            value=self.resolutions.index(saved_args.get('resolution',512)),
            scroll_exit = True,
            max_height=4,
        )
        self.center_crop = self.add_widget_intelligent(
            npyscreen.Checkbox,
            name="Center crop images before resizing to resolution",
            value=saved_args.get('center_crop',False)
        )
        self.mixed_precision = self.add_widget_intelligent(
            npyscreen.TitleSelectOne,
            name='Mixed Precision:',
            values=self.precisions,
            value=self.precisions.index(saved_args.get('mixed_precision','fp16')),
            max_height=4,
        )
        self.max_train_steps = self.add_widget_intelligent(
            npyscreen.TitleSlider,
            name='Max Training Steps:',
            out_of=10000,
            step=500,
            lowest=1,
            value=saved_args.get('max_train_steps',3000)
        )
        self.train_batch_size = self.add_widget_intelligent(
            npyscreen.TitleSlider,
            name='Batch Size (reduce if you run out of memory):',
            out_of=50,
            step=1,
            lowest=1,
            value=saved_args.get('train_batch_size',8),
        )
        self.learning_rate = self.add_widget_intelligent(
            npyscreen.TitleText,
            name="Learning Rate:",
            value=str(saved_args.get('learning_rate','5.0e-04'),)
        )
        self.scale_lr = self.add_widget_intelligent(
            npyscreen.Checkbox,
            name="Scale learning rate by number GPUs, steps and batch size",
            value=saved_args.get('scale_lr',True),
        )
        self.enable_xformers_memory_efficient_attention = self.add_widget_intelligent(
            npyscreen.Checkbox,
            name="Use xformers acceleration",
            value=saved_args.get('enable_xformers_memory_efficient_attention',False),
        )
        self.lr_scheduler = self.add_widget_intelligent(
            npyscreen.TitleSelectOne,
            name='Learning rate scheduler:',
            values = self.lr_schedulers,
            max_height=7,
            scroll_exit = True,
            value=self.lr_schedulers.index(saved_args.get('lr_scheduler','constant')),
        )
        self.gradient_accumulation_steps = self.add_widget_intelligent(
            npyscreen.TitleSlider,
            name='Gradient Accumulation Steps:',
            out_of=10,
            step=1,
            lowest=1,
            value=saved_args.get('gradient_accumulation_steps',4)
        )
        self.lr_warmup_steps = self.add_widget_intelligent(
            npyscreen.TitleSlider,
            name='Warmup Steps:',
            out_of=100,
            step=1,
            lowest=0,
            value=saved_args.get('lr_warmup_steps',0),
        )

    def initializer_changed(self):
        placeholder = self.placeholder_token.value
        self.prompt_token.value = f'(Trigger by using <{placeholder}> in your prompts)'
        self.train_data_dir.value = Path(Globals.root) / TRAINING_DATA / placeholder
        self.output_dir.value = Path(Globals.root) / TRAINING_DIR / placeholder
        self.resume_from_checkpoint.value = Path(self.output_dir.value).exists()
        
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

    def ok_cancel(self):
        sys.exit(0)

    def validate_field_values(self)->bool:
        bad_fields = []
        if self.model.value is None:
            bad_fields.append('Model Name must correspond to a known model in models.yaml')
        if not re.match('^[a-zA-Z0-9.-]+$',self.placeholder_token.value):
            bad_fields.append('Trigger term must only contain alphanumeric characters, the dot and hyphen')
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

    def marshall_arguments(self)->dict:
        args = dict()

        # the choices
        args.update(
            model = self.model_names[self.model.value[0]],
            resolution = self.resolutions[self.resolution.value[0]],
            lr_scheduler = self.lr_schedulers[self.lr_scheduler.value[0]],
            mixed_precision = self.precisions[self.mixed_precision.value[0]],
            learnable_property = self.learnable_properties[self.learnable_property.value[0]],
        )

        # all the strings and booleans
        for attr in ('initializer_token','placeholder_token','train_data_dir',
                     'output_dir','scale_lr','center_crop','enable_xformers_memory_efficient_attention'):
            args[attr] = getattr(self,attr).value
            
        # all the integers
        for attr in ('train_batch_size','gradient_accumulation_steps',
                     'max_train_steps','lr_warmup_steps'):
            args[attr] = int(getattr(self,attr).value)

        # the floats (just one)
        args.update(
            learning_rate = float(self.learning_rate.value)
        )

        # a special case
        if self.resume_from_checkpoint.value and Path(self.output_dir.value).exists():
            args['resume_from_checkpoint'] = 'latest'

        return args

class MyApplication(npyscreen.NPSAppManaged):
    def __init__(self, saved_args=None):
        super().__init__()
        self.ti_arguments=None
        self.saved_args=saved_args

    def onStart(self):
        npyscreen.setTheme(npyscreen.Themes.DefaultTheme)
        self.main = self.addForm('MAIN', textualInversionForm, name='Textual Inversion Settings', saved_args=self.saved_args)

def copy_to_embeddings_folder(args:dict):
    '''
    Copy learned_embeds.bin into the embeddings folder, and offer to
    delete the full model and checkpoints.
    '''
    source = Path(args['output_dir'],'learned_embeds.bin')
    dest_dir_name = args['placeholder_token'].strip('<>')
    destination = Path(Globals.root,'embeddings',dest_dir_name)
    os.makedirs(destination,exist_ok=True)
    print(f'>> Training completed. Copying learned_embeds.bin into {str(destination)}')
    shutil.copy(source,destination)
    if (input('Delete training logs and intermediate checkpoints? [y] ') or 'y').startswith(('y','Y')):
        shutil.rmtree(Path(args['output_dir']))
    else:
        print(f'>> Keeping {args["output_dir"]}')

def save_args(args:dict):
    '''
    Save the current argument values to an omegaconf file
    '''
    conf_file = Path(Globals.root) / TRAINING_DIR / CONF_FILE
    conf = OmegaConf.create(args)
    OmegaConf.save(config=conf, f=conf_file)

def previous_args()->dict:
    '''
    Get the previous arguments used.
    '''
    conf_file = Path(Globals.root) / TRAINING_DIR / CONF_FILE
    try:
        conf = OmegaConf.load(conf_file)
        conf['placeholder_token'] = conf['placeholder_token'].strip('<>')
    except:
        conf= None

    return conf

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

    saved_args = previous_args()
    myapplication = MyApplication(saved_args=saved_args)
    myapplication.run()
    
    from ldm.invoke.textual_inversion_training import do_textual_inversion_training
    if args := myapplication.ti_arguments:
        os.makedirs(args['output_dir'],exist_ok=True)
        
        # Automatically add angle brackets around the trigger
        if not re.match('^<.+>$',args['placeholder_token']):
            args['placeholder_token'] = f"<{args['placeholder_token']}>"

        args['only_save_embeds'] = True
        save_args(args)

        try:
            do_textual_inversion_training(**args)
            copy_to_embeddings_folder(args)
        except Exception as e:
            print('** An exception occurred during training. The exception was:')
            print(str(e))
            print('** DETAILS:')
            print(traceback.format_exc())
