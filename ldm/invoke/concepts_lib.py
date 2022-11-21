"""
Query and install embeddings from the HuggingFace SD Concepts Library
at https://huggingface.co/sd-concepts-library.

The interface is through the Concepts() object.
"""
import os
import re
from urllib import request
from huggingface_hub import HfFolder, hf_hub_url, ModelSearchArguments, ModelFilter, HfApi
from ldm.invoke.globals import Globals

class Concepts(object):
    def __init__(self, root=None):
        '''
        Initialize the Concepts object. May optionally pass a root directory.
        '''
        # this is a placeholder while HuggingFace fixes their server
        self.root = root or Globals.root
        self.concept_list_file = os.path.join(self.root,'configs/sd-concepts.txt')
        self.hf_api = HfApi()
        self.concept_list = None
        self.concepts_loaded = dict()
        self.triggers = dict()            # concept name to trigger phrase
        self.concept_names = dict()       # trigger phrase to concept name

    def list_concepts(self)->list:
        '''
        Return a list of all the concepts by name, without the 'sd-concepts-library' part.
        Note that this currently is reading the list from a static file, but it will retrieve
        from the HuggingFace API as soon as that is working again. (currently giving a 500 
        error!)
        '''
        if self.concept_list is not None:
            return self.concept_list

        self.concept_list = list()
        try:
            with open(self.concept_list_file,'r') as clf:
                concepts = clf.read().splitlines()
            for line in concepts:
                a = line.split('/')
                if len(a)<2:
                    continue
                library, model_name = a
                assert library == 'sd-concepts-library', f'** invalid line in {self.concept_list_file}: "{line}"'
                self.concept_list.append(model_name)
        except OSError as e:
            print(f'** An operating system error occurred while retrieving SD concepts: {str(e)}')
            return None
        return self.concept_list

    def get_concept_model_path(self, concept_name:str)->str:
        '''
        Returns the path to the 'learned_embeds.bin' file in
        the named concept. Returns None if invalid or cannot
        be downloaded.
        '''
        return self.get_concept_file(concept_name.lower(),'learned_embeds.bin')

    def concept_to_trigger(self, concept_name:str)->str:
        '''
        Given a concept name returns its trigger by looking in the
        "token_identifier.txt" file.
        '''
        if concept_name in self.triggers:
            return self.triggers[concept_name]
        file = self.get_concept_file(concept_name, 'token_identifier.txt')
        if not file:
            return None
        with open(file,'r') as f:
            trigger = f.readline()
            trigger = trigger.strip()
        self.triggers[concept_name] = trigger
        self.concept_names[trigger] = concept_name
        return trigger

    def trigger_to_concept(self, trigger:str)->str:
        '''
        Given a trigger phrase, maps it to the concept library name.
        Only works if concept_to_trigger() has previously been called
        on this library. There needs to be a persistent database for
        this.
        '''
        return f'<{self.concept_names.get(trigger,None)}>'

    def replace_triggers_with_concepts(self, prompt:str)->str:
        '''
        Given a prompt string that contains <trigger> tags, replace these
        tags with the concept name. The reason for this is so that the
        concept names get stored in the prompt metadata. There is no
        controlling of colliding triggers in the SD library, so it is
        better to store the concept name (unique) than the concept trigger
        (not necessarily unique!)
        '''
        def do_replace(match)->str:
            return self.trigger_to_concept(match.group(1)) or f'<{match.group(1)}>'
        return re.sub('(<[^>]+>)', do_replace, prompt)

    def replace_concepts_with_triggers(self, prompt:str)->str:
        '''
        Given a prompt string that contains <concept_name> tags, replace
        these tags with the appropriate trigger.
        '''
        def do_replace(match)->str:
            return self.concept_to_trigger(match.group(1)) or f'<{match.group(1)}>'
        return re.sub('<([^>]+)>', do_replace, prompt)

    def get_concept_file(self, concept_name:str, file_name:str='learned_embeds.bin')->str:
        if not self.concept_is_downloaded(concept_name):
            self.download_concept(concept_name)
        path = os.path.join(self._concept_path(concept_name), file_name)
        return path if os.path.exists(path) else None
        
    def concept_is_downloaded(self, concept_name)->bool:
        concept_directory = self._concept_path(concept_name)
        return os.path.exists(concept_directory)
        
    def download_concept(self,concept_name)->bool:
        repo_id = self._concept_id(concept_name)
        dest = self._concept_path(concept_name)

        os.makedirs(dest, exist_ok=True)
        succeeded = True
        bytes = 0

        def tally_download_size(chunk, size, total):
            nonlocal bytes
            if chunk==0:
                bytes += total
            
        print(f'>> Downloading {repo_id}...',end='')
        try:
            for file in ('README.md','learned_embeds.bin','token_identifier.txt','type_of_concept.txt'):
                url = hf_hub_url(repo_id, file)
                request.urlretrieve(url, os.path.join(dest,file),reporthook=tally_download_size)
        except Exception as e:
            print(f'failed to download {concept_name}/{file} ({str(e)})')
            return False
        print('...{:.2f}Kb'.format(bytes/1024))
        return succeeded

    def _concept_id(self, concept_name:str)->str:
        return f'sd-concepts-library/{concept_name}'

    def _concept_path(self, concept_name:str)->str:
        return os.path.join(self.root,'models','sd-concepts-library',concept_name)
