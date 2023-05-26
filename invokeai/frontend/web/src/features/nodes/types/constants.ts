import { FieldType, FieldUIConfig } from './types';

export const HANDLE_TOOLTIP_OPEN_DELAY = 500;

export const FIELD_TYPE_MAP: Record<string, FieldType> = {
  integer: 'integer',
  float: 'float',
  number: 'float',
  string: 'string',
  boolean: 'boolean',
  enum: 'enum',
  ImageField: 'image',
  LatentsField: 'latents',
  ConditioningField: 'conditioning',
  model: 'model',
  array: 'array',
  item: 'item',
  ColorField: 'color',
  ControlField: 'control',
  control: 'control',
};

const COLOR_TOKEN_VALUE = 500;

const getColorTokenCssVariable = (color: string) =>
  `var(--invokeai-colors-${color}-${COLOR_TOKEN_VALUE})`;

// @ts-ignore
// @ts-ignore
// @ts-ignore
export const FIELDS: Record<FieldType, FieldUIConfig> = {
  integer: {
    color: 'red',
    colorCssVar: getColorTokenCssVariable('red'),
    title: 'Integer',
    description: 'Integers are whole numbers, without a decimal point.',
  },
  float: {
    color: 'orange',
    colorCssVar: getColorTokenCssVariable('orange'),
    title: 'Float',
    description: 'Floats are numbers with a decimal point.',
  },
  string: {
    color: 'yellow',
    colorCssVar: getColorTokenCssVariable('yellow'),
    title: 'String',
    description: 'Strings are text.',
  },
  boolean: {
    color: 'green',
    colorCssVar: getColorTokenCssVariable('green'),
    title: 'Boolean',
    description: 'Booleans are true or false.',
  },
  enum: {
    color: 'blue',
    colorCssVar: getColorTokenCssVariable('blue'),
    title: 'Enum',
    description: 'Enums are values that may be one of a number of options.',
  },
  image: {
    color: 'purple',
    colorCssVar: getColorTokenCssVariable('purple'),
    title: 'Image',
    description: 'Images may be passed between nodes.',
  },
  latents: {
    color: 'pink',
    colorCssVar: getColorTokenCssVariable('pink'),
    title: 'Latents',
    description: 'Latents may be passed between nodes.',
  },
  conditioning: {
    color: 'cyan',
    colorCssVar: getColorTokenCssVariable('cyan'),
    title: 'Conditioning',
    description: 'Conditioning may be passed between nodes.',
  },
  control: {
    color: 'cyan',
    colorCssVar: getColorTokenCssVariable('cyan'), // TODO: no free color left
    title: 'Control',
    description: 'Control info passed between nodes.',
  },
  model: {
    color: 'teal',
    colorCssVar: getColorTokenCssVariable('teal'),
    title: 'Model',
    description: 'Models are models.',
  },
  array: {
    color: 'gray',
    colorCssVar: getColorTokenCssVariable('gray'),
    title: 'Array',
    description: 'TODO: Array type description.',
  },
  item: {
    color: 'gray',
    colorCssVar: getColorTokenCssVariable('gray'),
    title: 'Collection Item',
    description: 'TODO: Collection Item type description.',
  },
  color: {
    color: 'gray',
    colorCssVar: getColorTokenCssVariable('gray'),
    title: 'Color',
    description: 'A RGBA color.',
  },
};
