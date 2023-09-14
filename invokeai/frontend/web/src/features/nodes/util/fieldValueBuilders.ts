import { FieldType, InputFieldTemplate, InputFieldValue } from '../types/types';

const FIELD_VALUE_FALLBACK_MAP: {
  [key in FieldType]: InputFieldValue['value'];
} = {
  Any: undefined,
  enum: '',
  BoardField: undefined,
  boolean: false,
  BooleanCollection: [],
  BooleanPolymorphic: false,
  ClipField: undefined,
  Collection: [],
  CollectionItem: undefined,
  ColorCollection: [],
  ColorField: undefined,
  ColorPolymorphic: undefined,
  ConditioningCollection: [],
  ConditioningField: undefined,
  ConditioningPolymorphic: undefined,
  ControlCollection: [],
  ControlField: undefined,
  ControlNetModelField: undefined,
  ControlPolymorphic: undefined,
  DenoiseMaskField: undefined,
  float: 0,
  FloatCollection: [],
  FloatPolymorphic: 0,
  ImageCollection: [],
  ImageField: undefined,
  ImagePolymorphic: undefined,
  integer: 0,
  IntegerCollection: [],
  IntegerPolymorphic: 0,
  IPAdapterField: undefined,
  IPAdapterModelField: undefined,
  LatentsCollection: [],
  LatentsField: undefined,
  LatentsPolymorphic: undefined,
  MetadataItem: undefined,
  MetadataItemCollection: [],
  MetadataItemPolymorphic: undefined,
  MetadataDict: undefined,
  MetadataDictCollection: [],
  LoRAModelField: undefined,
  MainModelField: undefined,
  ONNXModelField: undefined,
  Scheduler: 'euler',
  SDXLMainModelField: undefined,
  SDXLRefinerModelField: undefined,
  string: '',
  StringCollection: [],
  StringPolymorphic: '',
  UNetField: undefined,
  VaeField: undefined,
  VaeModelField: undefined,
};

export const buildInputFieldValue = (
  id: string,
  template: InputFieldTemplate
): InputFieldValue => {
  // TODO: this should be `fieldValue: InputFieldValue`, but that introduces a TS issue I couldn't
  // resolve - for some reason, it doesn't like `template.type`, which is the discriminant for both
  // `InputFieldTemplate` union. It is (type-structurally) equal to the discriminant for the
  // `InputFieldValue` union, but TS doesn't seem to like it...
  const fieldValue = {
    id,
    name: template.name,
    type: template.type,
    label: '',
    fieldKind: 'input',
  } as InputFieldValue;

  fieldValue.value =
    template.default ?? FIELD_VALUE_FALLBACK_MAP[template.type];

  return fieldValue;
};
