"""Test class to just return reverse elements in a range"""

import paginate

class ReversePage(paginate.Page):
    #def __init__(self, *args, **kwargs):
        #super(ReversePage, self).__init__(*args, **kwargs)
    
    def __getitem__(self, key):
        #if isinstance(key, slice):
            #return list(reversed(self.collection))[key]
        #else:
        return list(reversed(self.collection))[key]
