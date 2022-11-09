"""Custom module."""
from ldm.generate import Generate

class ApiTextToImage(MethodView):
  @inject
  def __init__(self, pathBase, storage: ImageStorageService = Provide[Container.image_storage_service]):
    self.__pathRoot = os.path.abspath(os.path.join(os.path.dirname(__file__), pathBase))
    self.__storage = storage

  def get(self, dreamId):
    meta = self.__storage.getMetadata(dreamId)
    g       = Generate()
    outputs = g.txt2img("a unicorn in manhattan")
    j = {} if meta is None else meta.__dict__
    return j




