import { FieldType, FieldUIConfig } from './types';

export const HANDLE_TOOLTIP_OPEN_DELAY = 500;
export const COLOR_TOKEN_VALUE = 500;
export const NODE_WIDTH = 320;
export const NODE_MIN_WIDTH = 320;
export const DRAG_HANDLE_CLASSNAME = 'node-drag-handle';

export const IMAGE_FIELDS = ['ImageField', 'ImageCollection'];
export const FOOTER_FIELDS = IMAGE_FIELDS;

export const KIND_MAP = {
  input: 'inputs' as const,
  output: 'outputs' as const,
};

export const COLLECTION_TYPES: FieldType[] = [
  'Collection',
  'IntegerCollection',
  'BooleanCollection',
  'FloatCollection',
  'StringCollection',
  'ImageCollection',
  'LatentsCollection',
  'ConditioningCollection',
  'ControlCollection',
  'ColorCollection',
];

export const POLYMORPHIC_TYPES = [
  'IntegerPolymorphic',
  'BooleanPolymorphic',
  'FloatPolymorphic',
  'StringPolymorphic',
  'ImagePolymorphic',
  'LatentsPolymorphic',
  'ConditioningPolymorphic',
  'ControlPolymorphic',
  'ColorPolymorphic',
];

export const MODEL_TYPES = [
  'ControlNetModelField',
  'LoRAModelField',
  'MainModelField',
  'ONNXModelField',
  'SDXLMainModelField',
  'SDXLRefinerModelField',
  'VaeModelField',
  'UNetField',
  'VaeField',
  'ClipField',
];

export const COLLECTION_MAP = {
  integer: 'IntegerCollection',
  boolean: 'BooleanCollection',
  number: 'FloatCollection',
  float: 'FloatCollection',
  string: 'StringCollection',
  ImageField: 'ImageCollection',
  LatentsField: 'LatentsCollection',
  ConditioningField: 'ConditioningCollection',
  ControlField: 'ControlCollection',
  ColorField: 'ColorCollection',
};
export const isCollectionItemType = (
  itemType: string | undefined
): itemType is keyof typeof COLLECTION_MAP =>
  Boolean(itemType && itemType in COLLECTION_MAP);

export const SINGLE_TO_POLYMORPHIC_MAP = {
  integer: 'IntegerPolymorphic',
  boolean: 'BooleanPolymorphic',
  number: 'FloatPolymorphic',
  float: 'FloatPolymorphic',
  string: 'StringPolymorphic',
  ImageField: 'ImagePolymorphic',
  LatentsField: 'LatentsPolymorphic',
  ConditioningField: 'ConditioningPolymorphic',
  ControlField: 'ControlPolymorphic',
  ColorField: 'ColorPolymorphic',
};

export const POLYMORPHIC_TO_SINGLE_MAP = {
  IntegerPolymorphic: 'integer',
  BooleanPolymorphic: 'boolean',
  FloatPolymorphic: 'float',
  StringPolymorphic: 'string',
  ImagePolymorphic: 'ImageField',
  LatentsPolymorphic: 'LatentsField',
  ConditioningPolymorphic: 'ConditioningField',
  ControlPolymorphic: 'ControlField',
  ColorPolymorphic: 'ColorField',
};

export const isPolymorphicItemType = (
  itemType: string | undefined
): itemType is keyof typeof SINGLE_TO_POLYMORPHIC_MAP =>
  Boolean(itemType && itemType in SINGLE_TO_POLYMORPHIC_MAP);

export const FIELDS: Record<FieldType, FieldUIConfig> = {
  boolean: {
    color: 'green.500',
    description: 'Booleans are true or false.',
    title: 'Boolean',
  },
  BooleanCollection: {
    color: 'green.500',
    description: 'A collection of booleans.',
    title: 'Boolean Collection',
  },
  BooleanPolymorphic: {
    color: 'green.500',
    description: 'A collection of booleans.',
    title: 'Boolean Polymorphic',
  },
  ClipField: {
    color: 'green.500',
    description: 'Tokenizer and text_encoder submodels.',
    title: 'Clip',
  },
  Collection: {
    color: 'base.500',
    description: 'TODO',
    title: 'Collection',
  },
  CollectionItem: {
    color: 'base.500',
    description: 'TODO',
    title: 'Collection Item',
  },
  ColorCollection: {
    color: 'pink.300',
    description: 'A collection of colors.',
    title: 'Color Collection',
  },
  ColorField: {
    color: 'pink.300',
    description: 'A RGBA color.',
    title: 'Color',
  },
  ColorPolymorphic: {
    color: 'pink.300',
    description: 'A collection of colors.',
    title: 'Color Polymorphic',
  },
  ConditioningCollection: {
    color: 'cyan.500',
    description: 'Conditioning may be passed between nodes.',
    title: 'Conditioning Collection',
  },
  ConditioningField: {
    color: 'cyan.500',
    description: 'Conditioning may be passed between nodes.',
    title: 'Conditioning',
  },
  ConditioningPolymorphic: {
    color: 'cyan.500',
    description: 'Conditioning may be passed between nodes.',
    title: 'Conditioning Polymorphic',
  },
  ControlCollection: {
    color: 'teal.500',
    description: 'Control info passed between nodes.',
    title: 'Control Collection',
  },
  ControlField: {
    color: 'teal.500',
    description: 'Control info passed between nodes.',
    title: 'Control',
  },
  ControlNetModelField: {
    color: 'teal.500',
    description: 'TODO',
    title: 'ControlNet',
  },
  ControlPolymorphic: {
    color: 'teal.500',
    description: 'Control info passed between nodes.',
    title: 'Control Polymorphic',
  },
  DenoiseMaskField: {
    color: 'blue.300',
    description: 'Denoise Mask may be passed between nodes',
    title: 'Denoise Mask',
  },
  enum: {
    color: 'blue.500',
    description: 'Enums are values that may be one of a number of options.',
    title: 'Enum',
  },
  float: {
    color: 'orange.500',
    description: 'Floats are numbers with a decimal point.',
    title: 'Float',
  },
  FloatCollection: {
    color: 'orange.500',
    description: 'A collection of floats.',
    title: 'Float Collection',
  },
  FloatPolymorphic: {
    color: 'orange.500',
    description: 'A collection of floats.',
    title: 'Float Polymorphic',
  },
  ImageCollection: {
    color: 'purple.500',
    description: 'A collection of images.',
    title: 'Image Collection',
  },
  ImageField: {
    color: 'purple.500',
    description: 'Images may be passed between nodes.',
    title: 'Image',
  },
  ImagePolymorphic: {
    color: 'purple.500',
    description: 'A collection of images.',
    title: 'Image Polymorphic',
  },
  integer: {
    color: 'red.500',
    description: 'Integers are whole numbers, without a decimal point.',
    title: 'Integer',
  },
  IntegerCollection: {
    color: 'red.500',
    description: 'A collection of integers.',
    title: 'Integer Collection',
  },
  IntegerPolymorphic: {
    color: 'red.500',
    description: 'A collection of integers.',
    title: 'Integer Polymorphic',
  },
  LatentsCollection: {
    color: 'pink.500',
    description: 'Latents may be passed between nodes.',
    title: 'Latents Collection',
  },
  LatentsField: {
    color: 'pink.500',
    description: 'Latents may be passed between nodes.',
    title: 'Latents',
  },
  LatentsPolymorphic: {
    color: 'pink.500',
    description: 'Latents may be passed between nodes.',
    title: 'Latents Polymorphic',
  },
  LoRAModelField: {
    color: 'teal.500',
    description: 'TODO',
    title: 'LoRA',
  },
  MainModelField: {
    color: 'teal.500',
    description: 'TODO',
    title: 'Model',
  },
  ONNXModelField: {
    color: 'teal.500',
    description: 'ONNX model field.',
    title: 'ONNX Model',
  },
  Scheduler: {
    color: 'base.500',
    description: 'TODO',
    title: 'Scheduler',
  },
  SDXLMainModelField: {
    color: 'teal.500',
    description: 'SDXL model field.',
    title: 'SDXL Model',
  },
  SDXLRefinerModelField: {
    color: 'teal.500',
    description: 'TODO',
    title: 'Refiner Model',
  },
  string: {
    color: 'yellow.500',
    description: 'Strings are text.',
    title: 'String',
  },
  StringCollection: {
    color: 'yellow.500',
    description: 'A collection of strings.',
    title: 'String Collection',
  },
  StringPolymorphic: {
    color: 'yellow.500',
    description: 'A collection of strings.',
    title: 'String Polymorphic',
  },
  UNetField: {
    color: 'red.500',
    description: 'UNet submodel.',
    title: 'UNet',
  },
  VaeField: {
    color: 'blue.500',
    description: 'Vae submodel.',
    title: 'Vae',
  },
  VaeModelField: {
    color: 'teal.500',
    description: 'TODO',
    title: 'VAE',
  },
};
