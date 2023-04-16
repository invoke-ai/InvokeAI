import { getCSSVar } from '@chakra-ui/utils';
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

const COLOR_TOKEN_VALUE = 500;

const getColorTokenCssVariable = (color: string) =>
  `var(--invokeai-colors-${color}-${COLOR_TOKEN_VALUE})`;

export const FIELDS: Record<FieldType, FieldUIConfig> = {
  integer: {
    colorCssVar: getColorTokenCssVariable('red'),
    title: 'Integer',
    description: 'Integers are whole numbers, without a decimal point.',
  },
  float: {
    colorCssVar: getColorTokenCssVariable('orange'),
    title: 'Float',
    description: 'Floats are numbers with a decimal point.',
  },
  string: {
    colorCssVar: getColorTokenCssVariable('yellow'),
    title: 'String',
    description: 'Strings are text.',
  },
  boolean: {
    colorCssVar: getColorTokenCssVariable('green'),
    title: 'Boolean',
    description: 'Booleans are true or false.',
  },
  enum: {
    colorCssVar: getColorTokenCssVariable('blue'),
    title: 'Enum',
    description: 'Enums are values that may be one of a number of options.',
  },
  image: {
    colorCssVar: getColorTokenCssVariable('purple'),
    title: 'Image',
    description: 'Images may be passed between nodes.',
  },
  latents: {
    colorCssVar: getColorTokenCssVariable('pink'),
    title: 'Latents',
    description: 'Latents may be passed between nodes.',
  },
  model: {
    colorCssVar: getColorTokenCssVariable('teal'),
    title: 'Model',
    description: 'Models are models.',
  },
  array: {
    colorCssVar: getColorTokenCssVariable('gray'),
    title: 'Array',
    description: 'TODO: Array type description.',
  },
};
