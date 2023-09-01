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
  'FloatCollection',
  'StringCollection',
  'BooleanCollection',
  'ImageCollection',
];

export const FIELDS: Record<FieldType, FieldUIConfig> = {
  integer: {
    title: 'Integer',
    description: 'Integers are whole numbers, without a decimal point.',
    color: 'red.500',
  },
  float: {
    title: 'Float',
    description: 'Floats are numbers with a decimal point.',
    color: 'orange.500',
  },
  string: {
    title: 'String',
    description: 'Strings are text.',
    color: 'yellow.500',
  },
  boolean: {
    title: 'Boolean',
    color: 'green.500',
    description: 'Booleans are true or false.',
  },
  enum: {
    title: 'Enum',
    description: 'Enums are values that may be one of a number of options.',
    color: 'blue.500',
  },
  array: {
    title: 'Array',
    description: 'Enums are values that may be one of a number of options.',
    color: 'base.500',
  },
  ImageField: {
    title: 'Image',
    description: 'Images may be passed between nodes.',
    color: 'purple.500',
  },
  DenoiseMaskField: {
    title: 'Denoise Mask',
    description: 'Denoise Mask may be passed between nodes',
    color: 'base.500',
  },
  LatentsField: {
    title: 'Latents',
    description: 'Latents may be passed between nodes.',
    color: 'pink.500',
  },
  LatentsCollection: {
    title: 'Latents Collection',
    description: 'Latents may be passed between nodes.',
    color: 'pink.500',
  },
  ConditioningField: {
    color: 'cyan.500',
    title: 'Conditioning',
    description: 'Conditioning may be passed between nodes.',
  },
  ConditioningCollection: {
    color: 'cyan.500',
    title: 'Conditioning Collection',
    description: 'Conditioning may be passed between nodes.',
  },
  ImageCollection: {
    title: 'Image Collection',
    description: 'A collection of images.',
    color: 'base.300',
  },
  UNetField: {
    color: 'red.500',
    title: 'UNet',
    description: 'UNet submodel.',
  },
  ClipField: {
    color: 'green.500',
    title: 'Clip',
    description: 'Tokenizer and text_encoder submodels.',
  },
  VaeField: {
    color: 'blue.500',
    title: 'Vae',
    description: 'Vae submodel.',
  },
  ControlField: {
    color: 'cyan.500',
    title: 'Control',
    description: 'Control info passed between nodes.',
  },
  MainModelField: {
    color: 'teal.500',
    title: 'Model',
    description: 'TODO',
  },
  SDXLRefinerModelField: {
    color: 'teal.500',
    title: 'Refiner Model',
    description: 'TODO',
  },
  VaeModelField: {
    color: 'teal.500',
    title: 'VAE',
    description: 'TODO',
  },
  LoRAModelField: {
    color: 'teal.500',
    title: 'LoRA',
    description: 'TODO',
  },
  ControlNetModelField: {
    color: 'teal.500',
    title: 'ControlNet',
    description: 'TODO',
  },
  Scheduler: {
    color: 'base.500',
    title: 'Scheduler',
    description: 'TODO',
  },
  Collection: {
    color: 'base.500',
    title: 'Collection',
    description: 'TODO',
  },
  CollectionItem: {
    color: 'base.500',
    title: 'Collection Item',
    description: 'TODO',
  },
  ColorField: {
    title: 'Color',
    description: 'A RGBA color.',
    color: 'base.500',
  },
  BooleanCollection: {
    title: 'Boolean Collection',
    description: 'A collection of booleans.',
    color: 'green.500',
  },
  IntegerCollection: {
    title: 'Integer Collection',
    description: 'A collection of integers.',
    color: 'red.500',
  },
  FloatCollection: {
    color: 'orange.500',
    title: 'Float Collection',
    description: 'A collection of floats.',
  },
  ColorCollection: {
    color: 'base.500',
    title: 'Color Collection',
    description: 'A collection of colors.',
  },
  ONNXModelField: {
    color: 'base.500',
    title: 'ONNX Model',
    description: 'ONNX model field.',
  },
  SDXLMainModelField: {
    color: 'base.500',
    title: 'SDXL Model',
    description: 'SDXL model field.',
  },
  StringCollection: {
    color: 'yellow.500',
    title: 'String Collection',
    description: 'A collection of strings.',
  },
};
