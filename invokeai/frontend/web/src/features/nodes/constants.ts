import { FieldType, FieldUIConfig } from './types';

export const HANDLE_TOOLTIP_OPEN_DELAY = 500;

export const FIELD_TYPE_MAP: Record<string, FieldType> = {
  integer: 'integer',
  number: 'float',
  string: 'string',
  boolean: 'boolean',
  enum: 'enum',
  ImageField: 'image',
  LatentsField: 'latents',
  model: 'model',
  array: 'array',
};

export const FIELDS: Record<FieldType, FieldUIConfig> = {
  integer: {
    color: 'red',
    title: 'Integer',
    description: 'Integers are whole numbers, without a decimal point.',
  },
  float: {
    color: 'orange',
    title: 'Float',
    description: 'Floats are numbers with a decimal point.',
  },
  string: {
    color: 'yellow',
    title: 'String',
    description: 'Strings are text.',
  },
  boolean: {
    color: 'green',
    title: 'Boolean',
    description: 'Booleans are true or false.',
  },
  enum: {
    color: 'blue',
    title: 'Enum',
    description: 'Enums are values that may be one of a number of options.',
  },
  image: {
    color: 'purple',
    title: 'Image',
    description: 'Images may be passed between nodes.',
  },
  latents: {
    color: 'pink',
    title: 'Latents',
    description: 'Latents may be passed between nodes.',
  },
  model: {
    color: 'teal',
    title: 'Model',
    description: 'Models are models.',
  },
  array: {
    color: 'gray',
    title: 'Array',
    description: 'TODO: Array type description.',
  },
};
