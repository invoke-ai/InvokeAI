'''
This module defines a singleton object, "patchmatch" that
wraps the actual patchmatch object. It respects the global
"try_patchmatch" attribute, so that patchmatch loading can
be suppressed or deferred
'''
from ldm.invoke.globals import Globals
import numpy as  np

class PatchMatch:
    '''
    Thin class wrapper around the patchmatch function.
    '''

    patch_match = None
    tried_load:bool = False
    
    def __init__(self):
        super().__init__()

    @classmethod
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

    @classmethod
    def patchmatch_available(self)->bool:
        self._load_patch_match()
        return self.patch_match and self.patch_match.patchmatch_available

    @classmethod
    def inpaint(self,*args,**kwargs)->np.ndarray:
        if self.patchmatch_available():
            return self.patch_match.inpaint(*args,**kwargs)
