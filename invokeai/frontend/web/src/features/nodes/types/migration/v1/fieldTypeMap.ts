import { FieldType, StatefulFieldType } from '../../field';
import { FieldTypeV1 } from './workflowV1';

/**
 * Mapping of V1 field type strings to their *stateful* V2 field type counterparts.
 */
const FIELD_TYPE_V1_TO_STATEFUL_FIELD_TYPE_V2: {
  [key in FieldTypeV1]?: StatefulFieldType;
} = {
  BoardField: { name: 'BoardField', isCollection: false, isPolymorphic: false },
  boolean: { name: 'BooleanField', isCollection: false, isPolymorphic: false },
  BooleanCollection: {
    name: 'BooleanField',
    isCollection: true,
    isPolymorphic: false,
  },
  BooleanPolymorphic: {
    name: 'BooleanField',
    isCollection: false,
    isPolymorphic: true,
  },
  ColorField: { name: 'ColorField', isCollection: false, isPolymorphic: false },
  ColorCollection: {
    name: 'ColorField',
    isCollection: true,
    isPolymorphic: false,
  },
  ColorPolymorphic: {
    name: 'ColorField',
    isCollection: false,
    isPolymorphic: true,
  },
  ControlNetModelField: {
    name: 'ControlNetModelField',
    isCollection: false,
    isPolymorphic: false,
  },
  enum: { name: 'EnumField', isCollection: false, isPolymorphic: false },
  float: { name: 'FloatField', isCollection: false, isPolymorphic: false },
  FloatCollection: {
    name: 'FloatField',
    isCollection: true,
    isPolymorphic: false,
  },
  FloatPolymorphic: {
    name: 'FloatField',
    isCollection: false,
    isPolymorphic: true,
  },
  ImageCollection: {
    name: 'ImageField',
    isCollection: true,
    isPolymorphic: false,
  },
  ImageField: { name: 'ImageField', isCollection: false, isPolymorphic: false },
  ImagePolymorphic: {
    name: 'ImageField',
    isCollection: false,
    isPolymorphic: true,
  },
  integer: { name: 'IntegerField', isCollection: false, isPolymorphic: false },
  IntegerCollection: {
    name: 'IntegerField',
    isCollection: true,
    isPolymorphic: false,
  },
  IntegerPolymorphic: {
    name: 'IntegerField',
    isCollection: false,
    isPolymorphic: true,
  },
  IPAdapterModelField: {
    name: 'IPAdapterModelField',
    isCollection: false,
    isPolymorphic: false,
  },
  LoRAModelField: {
    name: 'LoRAModelField',
    isCollection: false,
    isPolymorphic: false,
  },
  MainModelField: {
    name: 'MainModelField',
    isCollection: false,
    isPolymorphic: false,
  },
  Scheduler: {
    name: 'SchedulerField',
    isCollection: false,
    isPolymorphic: false,
  },
  SDXLMainModelField: {
    name: 'SDXLMainModelField',
    isCollection: false,
    isPolymorphic: false,
  },
  SDXLRefinerModelField: {
    name: 'SDXLRefinerModelField',
    isCollection: false,
    isPolymorphic: false,
  },
  string: { name: 'StringField', isCollection: false, isPolymorphic: false },
  StringCollection: {
    name: 'StringField',
    isCollection: true,
    isPolymorphic: false,
  },
  StringPolymorphic: {
    name: 'StringField',
    isCollection: false,
    isPolymorphic: true,
  },
  T2IAdapterModelField: {
    name: 'T2IAdapterModelField',
    isCollection: false,
    isPolymorphic: false,
  },
  VaeModelField: {
    name: 'VAEModelField',
    isCollection: false,
    isPolymorphic: false,
  },
};

/**
 * Mapping of V1 field type strings to their *stateless* V2 field type counterparts.
 *
 * The type doesn't do what I want it to do.
 *
 * Ideally, the value of each propery would be a `FieldType` where `FieldType['name']` is not in
 * `StatefulFieldType['name']`, but this is hard to represent. That's because `FieldType['name']` is
 * actually widened to `string`, and TS's `Exclude<T,U>` doesn't work on `string`.
 *
 * There's probably some way to do it with conditionals and intersections but I can't figure it out.
 *
 * Thus, this object was manually edited to ensure it is correct.
 */
const FIELD_TYPE_V1_TO_STATELESS_FIELD_TYPE_V2: {
  [key in FieldTypeV1]?: FieldType;
} = {
  Any: { name: 'AnyField', isCollection: false, isPolymorphic: false },
  ClipField: { name: 'ClipField', isCollection: false, isPolymorphic: false },
  Collection: {
    name: 'CollectionField',
    isCollection: true,
    isPolymorphic: false,
  },
  CollectionItem: {
    name: 'CollectionItemField',
    isCollection: false,
    isPolymorphic: false,
  },
  ConditioningCollection: {
    name: 'ConditioningField',
    isCollection: true,
    isPolymorphic: false,
  },
  ConditioningField: {
    name: 'ConditioningField',
    isCollection: false,
    isPolymorphic: false,
  },
  ConditioningPolymorphic: {
    name: 'ConditioningField',
    isCollection: false,
    isPolymorphic: true,
  },
  ControlCollection: {
    name: 'ControlField',
    isCollection: true,
    isPolymorphic: false,
  },
  ControlField: {
    name: 'ControlField',
    isCollection: false,
    isPolymorphic: false,
  },
  ControlPolymorphic: {
    name: 'ControlField',
    isCollection: false,
    isPolymorphic: true,
  },
  DenoiseMaskField: {
    name: 'DenoiseMaskField',
    isCollection: false,
    isPolymorphic: false,
  },
  IPAdapterField: {
    name: 'IPAdapterField',
    isCollection: false,
    isPolymorphic: false,
  },
  IPAdapterCollection: {
    name: 'IPAdapterField',
    isCollection: true,
    isPolymorphic: false,
  },
  IPAdapterPolymorphic: {
    name: 'IPAdapterField',
    isCollection: false,
    isPolymorphic: true,
  },
  LatentsField: {
    name: 'LatentsField',
    isCollection: false,
    isPolymorphic: false,
  },
  LatentsCollection: {
    name: 'LatentsField',
    isCollection: true,
    isPolymorphic: false,
  },
  LatentsPolymorphic: {
    name: 'LatentsField',
    isCollection: false,
    isPolymorphic: true,
  },
  MetadataField: {
    name: 'MetadataField',
    isCollection: false,
    isPolymorphic: false,
  },
  MetadataCollection: {
    name: 'MetadataField',
    isCollection: true,
    isPolymorphic: false,
  },
  MetadataItemField: {
    name: 'MetadataItemField',
    isCollection: false,
    isPolymorphic: false,
  },
  MetadataItemCollection: {
    name: 'MetadataItemField',
    isCollection: true,
    isPolymorphic: false,
  },
  MetadataItemPolymorphic: {
    name: 'MetadataItemField',
    isCollection: false,
    isPolymorphic: true,
  },
  ONNXModelField: {
    name: 'ONNXModelField',
    isCollection: false,
    isPolymorphic: false,
  },
  T2IAdapterField: {
    name: 'T2IAdapterField',
    isCollection: false,
    isPolymorphic: false,
  },
  T2IAdapterCollection: {
    name: 'T2IAdapterField',
    isCollection: true,
    isPolymorphic: false,
  },
  T2IAdapterPolymorphic: {
    name: 'T2IAdapterField',
    isCollection: false,
    isPolymorphic: true,
  },
  UNetField: { name: 'UNetField', isCollection: false, isPolymorphic: false },
  VaeField: { name: 'VaeField', isCollection: false, isPolymorphic: false },
};

export const FIELD_TYPE_V1_TO_FIELD_TYPE_V2_MAPPING = {
  ...FIELD_TYPE_V1_TO_STATEFUL_FIELD_TYPE_V2,
  ...FIELD_TYPE_V1_TO_STATELESS_FIELD_TYPE_V2,
};
