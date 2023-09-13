import { FieldType, FieldUIConfig } from './types';
import i18n from 'i18next';

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
    description: i18n.t('nodes.booleanDescription'),
    title: i18n.t('nodes.boolean'),
  },
  BooleanCollection: {
    color: 'green.500',
    description: i18n.t('nodes.booleanCollectionDescription'),
    title: i18n.t('nodes.booleanCollection'),
  },
  BooleanPolymorphic: {
    color: 'green.500',
    description: i18n.t('nodes.booleanPolymorphicDescription'),
    title: i18n.t('nodes.booleanPolymorphic'),
  },
  ClipField: {
    color: 'green.500',
    description: i18n.t('nodes.clipFieldDescription'),
    title: i18n.t('nodes.clipField'),
  },
  Collection: {
    color: 'base.500',
    description: i18n.t('nodes.collectionDescription'),
    title: i18n.t('nodes.collection'),
  },
  CollectionItem: {
    color: 'base.500',
    description: i18n.t('nodes.collectionItemDescription'),
    title: i18n.t('nodes.collectionItem'),
  },
  ColorCollection: {
    color: 'pink.300',
    description: i18n.t('nodes.colorCollectionDescription'),
    title: i18n.t('nodes.colorCollection'),
  },
  ColorField: {
    color: 'pink.300',
    description: i18n.t('nodes.colorFieldDescription'),
    title: i18n.t('nodes.colorField'),
  },
  ColorPolymorphic: {
    color: 'pink.300',
    description: i18n.t('nodes.colorPolymorphicDescription'),
    title: i18n.t('nodes.colorPolymorphic'),
  },
  ConditioningCollection: {
    color: 'cyan.500',
    description: i18n.t('nodes.conditioningCollectionDescription'),
    title: i18n.t('nodes.conditioningCollection'),
  },
  ConditioningField: {
    color: 'cyan.500',
    description: i18n.t('nodes.conditioningFieldDescription'),
    title: i18n.t('nodes.conditioningField'),
  },
  ConditioningPolymorphic: {
    color: 'cyan.500',
    description: i18n.t('nodes.conditioningPolymorphicDescription'),
    title: i18n.t('nodes.conditioningPolymorphic'),
  },
  ControlCollection: {
    color: 'teal.500',
    description: i18n.t('nodes.controlCollectionDescription'),
    title: i18n.t('nodes.controlCollection'),
  },
  ControlField: {
    color: 'teal.500',
    description: i18n.t('nodes.controlFieldDescription'),
    title: i18n.t('nodes.controlField'),
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
    description: i18n.t('nodes.denoiseMaskFieldDescription'),
    title: i18n.t('nodes.denoiseMaskField'),
  },
  enum: {
    color: 'blue.500',
    description: i18n.t('nodes.enumDescription'),
    title: i18n.t('nodes.enum'),
  },
  float: {
    color: 'orange.500',
    description: i18n.t('nodes.floatDescription'),
    title: i18n.t('nodes.float'),
  },
  FloatCollection: {
    color: 'orange.500',
    description: i18n.t('nodes.floatCollectionDescription'),
    title: i18n.t('nodes.floatCollection'),
  },
  FloatPolymorphic: {
    color: 'orange.500',
    description: i18n.t('nodes.floatPolymorphicDescription'),
    title: i18n.t('nodes.floatPolymorphic'),
  },
  ImageCollection: {
    color: 'purple.500',
    description: i18n.t('nodes.imageCollectionDescription'),
    title: i18n.t('nodes.imageCollection'),
  },
  ImageField: {
    color: 'purple.500',
    description: i18n.t('nodes.imageFieldDescription'),
    title: i18n.t('nodes.imageField'),
  },
  ImagePolymorphic: {
    color: 'purple.500',
    description: i18n.t('nodes.imagePolymorphicDescription'),
    title: i18n.t('nodes.imagePolymorphic'),
  },
  integer: {
    color: 'red.500',
    description: i18n.t('nodes.integerDescription'),
    title: i18n.t('nodes.integer'),
  },
  IntegerCollection: {
    color: 'red.500',
    description: i18n.t('nodes.integerCollectionDescription'),
    title: i18n.t('nodes.integerCollection'),
  },
  IntegerPolymorphic: {
    color: 'red.500',
    description: i18n.t('nodes.integerPolymorphicDescription'),
    title: i18n.t('nodes.integerPolymorphic'),
  },
  LatentsCollection: {
    color: 'pink.500',
    description: i18n.t('nodes.latentsCollectionDescription'),
    title: i18n.t('nodes.latentsCollection'),
  },
  LatentsField: {
    color: 'pink.500',
    description: i18n.t('nodes.latentsFieldDescription'),
    title: i18n.t('nodes.latentsField'),
  },
  LatentsPolymorphic: {
    color: 'pink.500',
    description: i18n.t('nodes.latentsPolymorphicDescription'),
    title: i18n.t('nodes.latentsPolymorphic'),
  },
  LoRAModelField: {
    color: 'teal.500',
    description: i18n.t('nodes.loRAModelFieldDescription'),
    title: i18n.t('nodes.loRAModelField'),
  },
  MainModelField: {
    color: 'teal.500',
    description: i18n.t('nodes.mainModelFieldDescription'),
    title: i18n.t('nodes.mainModelField'),
  },
  ONNXModelField: {
    color: 'teal.500',
    description: i18n.t('nodes.oNNXModelFieldDescription'),
    title: i18n.t('nodes.oNNXModelField'),
  },
  Scheduler: {
    color: 'base.500',
    description: i18n.t('nodes.schedulerDescription'),
    title: i18n.t('nodes.scheduler'),
  },
  SDXLMainModelField: {
    color: 'teal.500',
    description: i18n.t('nodes.sDXLMainModelFieldDescription'),
    title: i18n.t('nodes.sDXLMainModelField'),
  },
  SDXLRefinerModelField: {
    color: 'teal.500',
    description: i18n.t('nodes.sDXLRefinerModelFieldDescription'),
    title: i18n.t('nodes.sDXLRefinerModelField'),
  },
  string: {
    color: 'yellow.500',
    description: i18n.t('nodes.stringDescription'),
    title: i18n.t('nodes.string'),
  },
  StringCollection: {
    color: 'yellow.500',
    description: i18n.t('nodes.stringCollectionDescription'),
    title: i18n.t('nodes.stringCollection'),
  },
  StringPolymorphic: {
    color: 'yellow.500',
    description: i18n.t('nodes.stringPolymorphicDescription'),
    title: i18n.t('nodes.stringPolymorphic'),
  },
  UNetField: {
    color: 'red.500',
    description: i18n.t('nodes.uNetFieldDescription'),
    title: i18n.t('nodes.uNetField'),
  },
  VaeField: {
    color: 'blue.500',
    description: i18n.t('nodes.vaeFieldDescription'),
    title: i18n.t('nodes.vaeField'),
  },
  VaeModelField: {
    color: 'teal.500',
    description: i18n.t('nodes.vaeModelFieldDescription'),
    title: i18n.t('nodes.vaeModelField'),
  },
};
