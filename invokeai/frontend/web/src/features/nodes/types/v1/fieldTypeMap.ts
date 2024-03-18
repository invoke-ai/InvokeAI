import type { FieldType, StatefulFieldType } from 'features/nodes/types/field';

import type { FieldTypeV1 } from './workflowV1';

/**
 * Mapping of V1 field type strings to their *stateful* V2 field type counterparts.
 */
const FIELD_TYPE_V1_TO_STATEFUL_FIELD_TYPE_V2: {
  [key in FieldTypeV1]?: StatefulFieldType;
} = {
  BoardField: {
    name: 'BoardField',
    isCollection: false,
    isCollectionOrScalar: false,
  },
  boolean: {
    name: 'BooleanField',
    isCollection: false,
    isCollectionOrScalar: false,
  },
  BooleanCollection: {
    name: 'BooleanField',
    isCollection: true,
    isCollectionOrScalar: false,
  },
  BooleanPolymorphic: {
    name: 'BooleanField',
    isCollection: false,
    isCollectionOrScalar: true,
  },
  ColorField: {
    name: 'ColorField',
    isCollection: false,
    isCollectionOrScalar: false,
  },
  ColorCollection: {
    name: 'ColorField',
    isCollection: true,
    isCollectionOrScalar: false,
  },
  ColorPolymorphic: {
    name: 'ColorField',
    isCollection: false,
    isCollectionOrScalar: true,
  },
  ControlNetModelField: {
    name: 'ControlNetModelField',
    isCollection: false,
    isCollectionOrScalar: false,
  },
  enum: { name: 'EnumField', isCollection: false, isCollectionOrScalar: false },
  float: {
    name: 'FloatField',
    isCollection: false,
    isCollectionOrScalar: false,
  },
  FloatCollection: {
    name: 'FloatField',
    isCollection: true,
    isCollectionOrScalar: false,
  },
  FloatPolymorphic: {
    name: 'FloatField',
    isCollection: false,
    isCollectionOrScalar: true,
  },
  ImageCollection: {
    name: 'ImageField',
    isCollection: true,
    isCollectionOrScalar: false,
  },
  ImageField: {
    name: 'ImageField',
    isCollection: false,
    isCollectionOrScalar: false,
  },
  ImagePolymorphic: {
    name: 'ImageField',
    isCollection: false,
    isCollectionOrScalar: true,
  },
  integer: {
    name: 'IntegerField',
    isCollection: false,
    isCollectionOrScalar: false,
  },
  IntegerCollection: {
    name: 'IntegerField',
    isCollection: true,
    isCollectionOrScalar: false,
  },
  IntegerPolymorphic: {
    name: 'IntegerField',
    isCollection: false,
    isCollectionOrScalar: true,
  },
  IPAdapterModelField: {
    name: 'IPAdapterModelField',
    isCollection: false,
    isCollectionOrScalar: false,
  },
  LoRAModelField: {
    name: 'LoRAModelField',
    isCollection: false,
    isCollectionOrScalar: false,
  },
  MainModelField: {
    name: 'MainModelField',
    isCollection: false,
    isCollectionOrScalar: false,
  },
  Scheduler: {
    name: 'SchedulerField',
    isCollection: false,
    isCollectionOrScalar: false,
  },
  SDXLMainModelField: {
    name: 'SDXLMainModelField',
    isCollection: false,
    isCollectionOrScalar: false,
  },
  SDXLRefinerModelField: {
    name: 'SDXLRefinerModelField',
    isCollection: false,
    isCollectionOrScalar: false,
  },
  string: {
    name: 'StringField',
    isCollection: false,
    isCollectionOrScalar: false,
  },
  StringCollection: {
    name: 'StringField',
    isCollection: true,
    isCollectionOrScalar: false,
  },
  StringPolymorphic: {
    name: 'StringField',
    isCollection: false,
    isCollectionOrScalar: true,
  },
  T2IAdapterModelField: {
    name: 'T2IAdapterModelField',
    isCollection: false,
    isCollectionOrScalar: false,
  },
  VaeModelField: {
    name: 'VAEModelField',
    isCollection: false,
    isCollectionOrScalar: false,
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
  Any: { name: 'AnyField', isCollection: false, isCollectionOrScalar: false },
  ClipField: {
    name: 'ClipField',
    isCollection: false,
    isCollectionOrScalar: false,
  },
  Collection: {
    name: 'CollectionField',
    isCollection: true,
    isCollectionOrScalar: false,
  },
  CollectionItem: {
    name: 'CollectionItemField',
    isCollection: false,
    isCollectionOrScalar: false,
  },
  ConditioningCollection: {
    name: 'ConditioningField',
    isCollection: true,
    isCollectionOrScalar: false,
  },
  ConditioningField: {
    name: 'ConditioningField',
    isCollection: false,
    isCollectionOrScalar: false,
  },
  ConditioningPolymorphic: {
    name: 'ConditioningField',
    isCollection: false,
    isCollectionOrScalar: true,
  },
  ControlCollection: {
    name: 'ControlField',
    isCollection: true,
    isCollectionOrScalar: false,
  },
  ControlField: {
    name: 'ControlField',
    isCollection: false,
    isCollectionOrScalar: false,
  },
  ControlPolymorphic: {
    name: 'ControlField',
    isCollection: false,
    isCollectionOrScalar: true,
  },
  DenoiseMaskField: {
    name: 'DenoiseMaskField',
    isCollection: false,
    isCollectionOrScalar: false,
  },
  IPAdapterField: {
    name: 'IPAdapterField',
    isCollection: false,
    isCollectionOrScalar: false,
  },
  IPAdapterCollection: {
    name: 'IPAdapterField',
    isCollection: true,
    isCollectionOrScalar: false,
  },
  IPAdapterPolymorphic: {
    name: 'IPAdapterField',
    isCollection: false,
    isCollectionOrScalar: true,
  },
  LatentsField: {
    name: 'LatentsField',
    isCollection: false,
    isCollectionOrScalar: false,
  },
  LatentsCollection: {
    name: 'LatentsField',
    isCollection: true,
    isCollectionOrScalar: false,
  },
  LatentsPolymorphic: {
    name: 'LatentsField',
    isCollection: false,
    isCollectionOrScalar: true,
  },
  MetadataField: {
    name: 'MetadataField',
    isCollection: false,
    isCollectionOrScalar: false,
  },
  MetadataCollection: {
    name: 'MetadataField',
    isCollection: true,
    isCollectionOrScalar: false,
  },
  MetadataItemField: {
    name: 'MetadataItemField',
    isCollection: false,
    isCollectionOrScalar: false,
  },
  MetadataItemCollection: {
    name: 'MetadataItemField',
    isCollection: true,
    isCollectionOrScalar: false,
  },
  MetadataItemPolymorphic: {
    name: 'MetadataItemField',
    isCollection: false,
    isCollectionOrScalar: true,
  },
  T2IAdapterField: {
    name: 'T2IAdapterField',
    isCollection: false,
    isCollectionOrScalar: false,
  },
  T2IAdapterCollection: {
    name: 'T2IAdapterField',
    isCollection: true,
    isCollectionOrScalar: false,
  },
  T2IAdapterPolymorphic: {
    name: 'T2IAdapterField',
    isCollection: false,
    isCollectionOrScalar: true,
  },
  UNetField: {
    name: 'UNetField',
    isCollection: false,
    isCollectionOrScalar: false,
  },
  VaeField: {
    name: 'VaeField',
    isCollection: false,
    isCollectionOrScalar: false,
  },
};

export const FIELD_TYPE_V1_TO_FIELD_TYPE_V2_MAPPING = {
  ...FIELD_TYPE_V1_TO_STATEFUL_FIELD_TYPE_V2,
  ...FIELD_TYPE_V1_TO_STATELESS_FIELD_TYPE_V2,
};
