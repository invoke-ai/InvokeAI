# # Copyright (c) 2022 Kyle Schouviller (https://github.com/kyle0654)

# from marshmallow import EXCLUDE, Schema, SchemaOpts, ValidationError, fields, INCLUDE, validates_schema
# from marshmallow.fields import Field
# from marshmallow.validate import OneOf, Range
# from marshmallow_oneofschema import OneOfSchema
# from graphlib import TopologicalSorter, CycleError


# from ldm.dream.app.invocations.baseinvocation import InvocationABC



# class BaseInvoker(BaseModel):
#     """The base invoker model from which all models must derive"""
#     @classmethod
#     def get_subclasses(cls):
#         return tuple(cls.__subclasses__())

#     @classmethod
#     def get_subclasses_map(cls):
#         # Get the type strings out of the literals and into a dictionary
#         return dict(map(lambda t: (get_args(get_type_hints(t)['type'])[0], t),cls.__subclasses__()))
        
#     @abstractmethod
#     def invoke(self) -> str:
#         pass

#     id: str = Field(description="The id of this node. Must be unique among all nodes.")







# class ImageField(Field):
#   """Used to mark image inputs/outputs. Only usable with links."""
#   def __init__(self, *args, **kwargs):
#     kwargs.setdefault('load_default', None)
#     kwargs.setdefault('dump_default', None)
#     kwargs.setdefault('allow_none', True)
#     super().__init__(*args, **kwargs)


# class MissingInvocation(Exception):
#     ...


# class NotAnInvocation(Exception):
#     ...


# class InvocationSchemaOpts(SchemaOpts):
#   """Adds outputs to Meta options to enable link validation."""

#   def __init__(self, meta, **kwargs):
#     SchemaOpts.__init__(self, meta, **kwargs)
#     if not getattr(meta, 'register', True):
#         return;

#     self.outputs = getattr(meta, 'outputs', {})
#     self.type = getattr(meta, 'type', None)
#     self.invokes = getattr(meta, 'invokes', None)

#     if not self.invokes:
#         raise MissingInvocation('Must provide an InvocationSchema through "invokes" on class Meta')
#     if not issubclass(self.invokes, InvocationABC):
#         raise NotAnInvocation('Invocation provided through "invokes" does not derive from InvocationABC')


# class InvocationSchemaBase(Schema):
#   OPTIONS_CLASS = InvocationSchemaOpts

#   class Meta:
#       register = False

#   id = fields.String(required=True)
#   type = fields.String(required=True)





# # class GFPGANSchema(InvocationSchema):
# #   """Face restoration"""
# #   class Meta:
# #     type = 'gfpgan'
# #     outputs = {
# #       'image': ImageField()
# #     }

# #   image = ImageField()
# #   strength = fields.Float(load_default=0.75, validate=Range(0.0, 1.0, min_inclusive=False, max_inclusive=True))


# # # # TODO: Fill this out
# # # class EmbiggenSchema(ProcessorSchema):
# # #   """Embiggen"""
# # #   embiggen = fields.Raw()
# # #   embiggen_tiles = fields.Raw()





# class ProcessorsSchema(OneOfSchema):
#   """OneOfSchema that can load all processors if their 'type' matches"""
#   @staticmethod
#   def __all_subclasses(cls):
#     return set(cls.__subclasses__()).union(
#         [s for c in cls.__subclasses__() for s in ProcessorsSchema.__all_subclasses(c)])

#   def __init__(self, *args, **kwargs):
#     # We define this in init instead of class so it catches processor schemas defined outside this file
#     self.type_schemas = dict(map(lambda s: (s.Meta.type, s), ProcessorsSchema.__all_subclasses(InvocationSchemaBase)))
#     self.type_field_remove = False
#     super().__init__(*args, **kwargs)

#   def get_schema(self, node):
#       """Gets the invocation schema provided a node to invoke"""
#       return self.type_schemas[node['type']]


# class NodeFieldSchema(Schema):
#     id = fields.String(required=True)
#     field = fields.String(required=True)


# class LinkSchema(Schema):
#     from_node = fields.Nested(NodeFieldSchema, required=True)
#     to_node = fields.Nested(NodeFieldSchema, required=True)


# class DreamMapSchema(Schema):
#   """The map for processing"""
#   nodes = fields.Nested(ProcessorsSchema, many=True)
#   links = fields.Nested(LinkSchema, many=True)

#   # TODO: Make a map class with helper methods on it
#   @staticmethod
#   def build_graph(dreamMap):
#       graph_nodes = nodes_dict = dict({dreamMap['nodes'][i]['id']: set() for i in range(0, len(dreamMap['nodes']))})

#       # Link all predecessors
#       for link in dreamMap['links']:
#           graph_nodes[link['to_node']['id']].add(link['from_node']['id'])

#       graph = TopologicalSorter(graph_nodes)
#       return graph
  
#   @staticmethod
#   def get_node_input_links(dreamMap, node):
#       return filter(lambda l: l['to_node']['id'] == node['id'], dreamMap['links'])


#   @staticmethod
#   def get_node(dreamMap, id: str):
#       for node in dreamMap['nodes']:
#           if (node['id'] == id):
#               return node
#       return None


#   # validate_schema (validate node ids, validate all links and their types)
#   @validates_schema
#   def validate_nodes_and_links(self, data, **kwargs):
#       # Check for duplicate node ids
#       ids_list = list(map(lambda n: n['id'], data['nodes']))
#       ids_set = set(ids_list)
#       if len(ids_list) != len(ids_set):
#           raise ValidationError('All node ids must be unique')

#       nodes_dict = dict({data['nodes'][i]['id']: data['nodes'][i] for i in range(0, len(data['nodes']))})

#       # Validate all links
#       errors = {}
#       for i in range(len(data['links'])):
#           link = data['links'][i]
#           link_errors = False

#           # Ensure node ids both exist
#           from_id = link['from_node']['id']
#           to_id = link['to_node']['id']

#           if from_id not in ids_set:
#               errors['links'] = [f'from_node.id {from_id} does not match any node id']
#               link_errors = True

#           if to_id not in ids_set:
#               errors['links'] = [f'to_node.id {to_id} does not match any node id']
#               link_errors = True

#           if from_id == to_id:
#               errors['links'] = [f'node {from_id} must not link to itself']
#               link_errors = True
          
#           if link_errors:
#               continue

#           # Get node types
#           ps = ProcessorsSchema()

#           from_node = nodes_dict[from_id]
#           to_node = nodes_dict[to_id]

#           from_type = from_node['type']
#           to_type = to_node['type']

#           from_node_type = ps.type_schemas[from_type]
#           to_node_type = ps.type_schemas[to_type]

#           # Ensure field types match
#           from_field = link['from_node']['field']
#           to_field = link['to_node']['field']

#           # Get outputs of from node          
#           from_node_meta = getattr(from_node_type, 'Meta', None)
#           from_node_outputs = None if not from_node_meta else getattr(from_node_meta, 'outputs', None)
#           from_node_field = None if not from_node_outputs else (from_node_outputs.get(from_field) or None)
#           # Get inputs of to node
#           to_node_field = to_node_type().fields.get(to_field) or None

#           if not from_node_field:
#               errors['links'] = [f'field {from_field} not found on node {from_id} of type {from_type}']
#           if not to_node_field:
#               errors['links'] = [f'field {to_field} not found on node {to_id} of type {to_type}']
          
#           if from_node_field and to_node_field:
#               if type(from_node_field) != type(to_node_field):
#                   errors['links'] = [f'field {from_field} of node {from_id} and type {from_type} does not match type {to_type} of node {to_id} field {to_field}']

#       if errors:
#           raise ValidationError(errors)
        
#       # Validate that this is a directed graph (no cycles)
#       ts = DreamMapSchema.build_graph(data)
#       try:
#           ts.prepare()
#       except CycleError:
#           raise ValidationError('Node graph must not have any cycles (loops)')
