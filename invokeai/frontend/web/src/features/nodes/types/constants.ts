import {
  FieldType,
  FieldTypeMap,
  FieldTypeMapWithNumber,
  FieldUIConfig,
} from './types';
import { t } from 'i18next';

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
  'T2IAdapterCollection',
  'IPAdapterCollection',
  'MetadataItemCollection',
  'MetadataCollection',
];

export const POLYMORPHIC_TYPES: FieldType[] = [
  'IntegerPolymorphic',
  'BooleanPolymorphic',
  'FloatPolymorphic',
  'StringPolymorphic',
  'ImagePolymorphic',
  'LatentsPolymorphic',
  'ConditioningPolymorphic',
  'ControlPolymorphic',
  'ColorPolymorphic',
  'T2IAdapterPolymorphic',
  'IPAdapterPolymorphic',
  'MetadataItemPolymorphic',
];

export const MODEL_TYPES: FieldType[] = [
  'IPAdapterModelField',
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
  'T2IAdapterModelField',
  'IPAdapterModelField',
];

export const COLLECTION_MAP: FieldTypeMapWithNumber = {
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
  T2IAdapterField: 'T2IAdapterCollection',
  IPAdapterField: 'IPAdapterCollection',
  MetadataItemField: 'MetadataItemCollection',
  MetadataField: 'MetadataCollection',
};
export const isCollectionItemType = (
  itemType: string | undefined
): itemType is keyof typeof COLLECTION_MAP =>
  Boolean(itemType && itemType in COLLECTION_MAP);

export const SINGLE_TO_POLYMORPHIC_MAP: FieldTypeMapWithNumber = {
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
  T2IAdapterField: 'T2IAdapterPolymorphic',
  IPAdapterField: 'IPAdapterPolymorphic',
  MetadataItemField: 'MetadataItemPolymorphic',
};

export const POLYMORPHIC_TO_SINGLE_MAP: FieldTypeMap = {
  IntegerPolymorphic: 'integer',
  BooleanPolymorphic: 'boolean',
  FloatPolymorphic: 'float',
  StringPolymorphic: 'string',
  ImagePolymorphic: 'ImageField',
  LatentsPolymorphic: 'LatentsField',
  ConditioningPolymorphic: 'ConditioningField',
  ControlPolymorphic: 'ControlField',
  ColorPolymorphic: 'ColorField',
  T2IAdapterPolymorphic: 'T2IAdapterField',
  IPAdapterPolymorphic: 'IPAdapterField',
  MetadataItemPolymorphic: 'MetadataItemField',
};

export const TYPES_WITH_INPUT_COMPONENTS: FieldType[] = [
  'string',
  'StringPolymorphic',
  'boolean',
  'BooleanPolymorphic',
  'integer',
  'float',
  'FloatPolymorphic',
  'IntegerPolymorphic',
  'enum',
  'ImageField',
  'ImagePolymorphic',
  'MainModelField',
  'SDXLRefinerModelField',
  'VaeModelField',
  'LoRAModelField',
  'ControlNetModelField',
  'ColorField',
  'SDXLMainModelField',
  'Scheduler',
  'IPAdapterModelField',
  'BoardField',
  'T2IAdapterModelField',
];

export const isPolymorphicItemType = (
  itemType: string | undefined
): itemType is keyof typeof SINGLE_TO_POLYMORPHIC_MAP =>
  Boolean(itemType && itemType in SINGLE_TO_POLYMORPHIC_MAP);

export const FIELDS: Record<FieldType, FieldUIConfig> = {
  Any: {
    color: 'gray.500',
    description: 'Any field type is accepted.',
    title: 'Any',
  },
  MetadataField: {
    color: 'gray.500',
    description: 'A metadata dict.',
    title: 'Metadata Dict',
  },
  MetadataCollection: {
    color: 'gray.500',
    description: 'A collection of metadata dicts.',
    title: 'Metadata Dict Collection',
  },
  MetadataItemField: {
    color: 'gray.500',
    description: 'A metadata item.',
    title: 'Metadata Item',
  },
  MetadataItemCollection: {
    color: 'gray.500',
    description: 'Any field type is accepted.',
    title: 'Metadata Item Collection',
  },
  MetadataItemPolymorphic: {
    color: 'gray.500',
    description:
      'MetadataItem or MetadataItemCollection field types are accepted.',
    title: 'Metadata Item Polymorphic',
  },
  boolean: {
    color: 'green.500',
    description: t('nodes.booleanDescription'),
    title: t('nodes.boolean'),
  },
  BooleanCollection: {
    color: 'green.500',
    description: t('nodes.booleanCollectionDescription'),
    title: t('nodes.booleanCollection'),
  },
  BooleanPolymorphic: {
    color: 'green.500',
    description: t('nodes.booleanPolymorphicDescription'),
    title: t('nodes.booleanPolymorphic'),
  },
  ClipField: {
    color: 'green.500',
    description: t('nodes.clipFieldDescription'),
    title: t('nodes.clipField'),
  },
  Collection: {
    color: 'base.500',
    description: t('nodes.collectionDescription'),
    title: t('nodes.collection'),
  },
  CollectionItem: {
    color: 'base.500',
    description: t('nodes.collectionItemDescription'),
    title: t('nodes.collectionItem'),
  },
  ColorCollection: {
    color: 'pink.300',
    description: t('nodes.colorCollectionDescription'),
    title: t('nodes.colorCollection'),
  },
  ColorField: {
    color: 'pink.300',
    description: t('nodes.colorFieldDescription'),
    title: t('nodes.colorField'),
  },
  ColorPolymorphic: {
    color: 'pink.300',
    description: t('nodes.colorPolymorphicDescription'),
    title: t('nodes.colorPolymorphic'),
  },
  ConditioningCollection: {
    color: 'cyan.500',
    description: t('nodes.conditioningCollectionDescription'),
    title: t('nodes.conditioningCollection'),
  },
  ConditioningField: {
    color: 'cyan.500',
    description: t('nodes.conditioningFieldDescription'),
    title: t('nodes.conditioningField'),
  },
  ConditioningPolymorphic: {
    color: 'cyan.500',
    description: t('nodes.conditioningPolymorphicDescription'),
    title: t('nodes.conditioningPolymorphic'),
  },
  ControlCollection: {
    color: 'teal.500',
    description: t('nodes.controlCollectionDescription'),
    title: t('nodes.controlCollection'),
  },
  ControlField: {
    color: 'teal.500',
    description: t('nodes.controlFieldDescription'),
    title: t('nodes.controlField'),
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
    description: t('nodes.denoiseMaskFieldDescription'),
    title: t('nodes.denoiseMaskField'),
  },
  enum: {
    color: 'blue.500',
    description: t('nodes.enumDescription'),
    title: t('nodes.enum'),
  },
  float: {
    color: 'orange.500',
    description: t('nodes.floatDescription'),
    title: t('nodes.float'),
  },
  FloatCollection: {
    color: 'orange.500',
    description: t('nodes.floatCollectionDescription'),
    title: t('nodes.floatCollection'),
  },
  FloatPolymorphic: {
    color: 'orange.500',
    description: t('nodes.floatPolymorphicDescription'),
    title: t('nodes.floatPolymorphic'),
  },
  ImageCollection: {
    color: 'purple.500',
    description: t('nodes.imageCollectionDescription'),
    title: t('nodes.imageCollection'),
  },
  ImageField: {
    color: 'purple.500',
    description: t('nodes.imageFieldDescription'),
    title: t('nodes.imageField'),
  },
  BoardField: {
    color: 'purple.500',
    description: t('nodes.imageFieldDescription'),
    title: t('nodes.imageField'),
  },
  ImagePolymorphic: {
    color: 'purple.500',
    description: t('nodes.imagePolymorphicDescription'),
    title: t('nodes.imagePolymorphic'),
  },
  integer: {
    color: 'red.500',
    description: t('nodes.integerDescription'),
    title: t('nodes.integer'),
  },
  IntegerCollection: {
    color: 'red.500',
    description: t('nodes.integerCollectionDescription'),
    title: t('nodes.integerCollection'),
  },
  IntegerPolymorphic: {
    color: 'red.500',
    description: t('nodes.integerPolymorphicDescription'),
    title: t('nodes.integerPolymorphic'),
  },
  IPAdapterCollection: {
    color: 'teal.500',
    description: t('nodes.ipAdapterCollectionDescription'),
    title: t('nodes.ipAdapterCollection'),
  },
  IPAdapterField: {
    color: 'teal.500',
    description: t('nodes.ipAdapterDescription'),
    title: t('nodes.ipAdapter'),
  },
  IPAdapterModelField: {
    color: 'teal.500',
    description: t('nodes.ipAdapterModelDescription'),
    title: t('nodes.ipAdapterModel'),
  },
  IPAdapterPolymorphic: {
    color: 'teal.500',
    description: t('nodes.ipAdapterPolymorphicDescription'),
    title: t('nodes.ipAdapterPolymorphic'),
  },
  LatentsCollection: {
    color: 'pink.500',
    description: t('nodes.latentsCollectionDescription'),
    title: t('nodes.latentsCollection'),
  },
  LatentsField: {
    color: 'pink.500',
    description: t('nodes.latentsFieldDescription'),
    title: t('nodes.latentsField'),
  },
  LatentsPolymorphic: {
    color: 'pink.500',
    description: t('nodes.latentsPolymorphicDescription'),
    title: t('nodes.latentsPolymorphic'),
  },
  LoRAModelField: {
    color: 'teal.500',
    description: t('nodes.loRAModelFieldDescription'),
    title: t('nodes.loRAModelField'),
  },
  MainModelField: {
    color: 'teal.500',
    description: t('nodes.mainModelFieldDescription'),
    title: t('nodes.mainModelField'),
  },
  ONNXModelField: {
    color: 'teal.500',
    description: t('nodes.oNNXModelFieldDescription'),
    title: t('nodes.oNNXModelField'),
  },
  Scheduler: {
    color: 'base.500',
    description: t('nodes.schedulerDescription'),
    title: t('nodes.scheduler'),
  },
  SDXLMainModelField: {
    color: 'teal.500',
    description: t('nodes.sDXLMainModelFieldDescription'),
    title: t('nodes.sDXLMainModelField'),
  },
  SDXLRefinerModelField: {
    color: 'teal.500',
    description: t('nodes.sDXLRefinerModelFieldDescription'),
    title: t('nodes.sDXLRefinerModelField'),
  },
  string: {
    color: 'yellow.500',
    description: t('nodes.stringDescription'),
    title: t('nodes.string'),
  },
  StringCollection: {
    color: 'yellow.500',
    description: t('nodes.stringCollectionDescription'),
    title: t('nodes.stringCollection'),
  },
  StringPolymorphic: {
    color: 'yellow.500',
    description: t('nodes.stringPolymorphicDescription'),
    title: t('nodes.stringPolymorphic'),
  },
  T2IAdapterCollection: {
    color: 'teal.500',
    description: t('nodes.t2iAdapterCollectionDescription'),
    title: t('nodes.t2iAdapterCollection'),
  },
  T2IAdapterField: {
    color: 'teal.500',
    description: t('nodes.t2iAdapterFieldDescription'),
    title: t('nodes.t2iAdapterField'),
  },
  T2IAdapterModelField: {
    color: 'teal.500',
    description: 'TODO',
    title: 'T2I-Adapter',
  },
  T2IAdapterPolymorphic: {
    color: 'teal.500',
    description: 'T2I-Adapter info passed between nodes.',
    title: 'T2I-Adapter Polymorphic',
  },
  UNetField: {
    color: 'red.500',
    description: t('nodes.uNetFieldDescription'),
    title: t('nodes.uNetField'),
  },
  VaeField: {
    color: 'blue.500',
    description: t('nodes.vaeFieldDescription'),
    title: t('nodes.vaeField'),
  },
  VaeModelField: {
    color: 'teal.500',
    description: t('nodes.vaeModelFieldDescription'),
    title: t('nodes.vaeModelField'),
  },
};
