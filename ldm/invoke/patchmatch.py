'''
This module defines a singleton object, "patchmatch" that
wraps the actual patchmatch object. It respects the global
"try_patchmatch" attribute, so that patchmatch loading can
be suppressed or deferred
'''
from ldm.invoke.globals import Globals
import numpy as  np

class Patchmatch (object):
    '''
    Thin class wrapper around the patchmatch function.
    '''
    
    def __init__(self):
        self.patch_match = None
        self.tried_load:bool = False
        super().__init__()

    def _load_patch_match(self):
        if self.tried_load:
            return
        if Globals.try_patchmatch:
            from patchmatch import patch_match as pm
            if pm.patchmatch_available:
                print('>> Patchmatch initialized')
            else:
                print('>> Patchmatch not loaded (nonfatal)')
            self.patch_match = pm
        else:
            print('>> Patchmatch loading disabled')
        self.tried_load = True

    def patchmatch_available(self)->bool:
        self._load_patch_match()
        return self.patch_match and self.patch_match.patchmatch_available

    def inpaint(self,*args,**kwargs)->np.ndarray:
        if self.patchmatch_available():
            return self.patch_match.inpaint(*args,**kwargs)


patchmatch=Patchmatch()
