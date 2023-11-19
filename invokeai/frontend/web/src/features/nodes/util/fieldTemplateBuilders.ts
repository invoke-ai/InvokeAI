import {
  isArray,
  isBoolean,
  isInteger,
  isNumber,
  isString,
  startCase,
} from 'lodash-es';
import { OpenAPIV3_1 } from 'openapi-types';
import { ControlField } from 'services/api/types';
import {
  COLLECTION_MAP,
  POLYMORPHIC_TYPES,
  SINGLE_TO_POLYMORPHIC_MAP,
  isCollectionItemType,
  isPolymorphicItemType,
} from '../types/constants';
import {
  AnyInputFieldTemplate,
  BoardInputFieldTemplate,
  BooleanCollectionInputFieldTemplate,
  BooleanInputFieldTemplate,
  BooleanPolymorphicInputFieldTemplate,
  ClipInputFieldTemplate,
  CollectionInputFieldTemplate,
  CollectionItemInputFieldTemplate,
  ColorCollectionInputFieldTemplate,
  ColorInputFieldTemplate,
  ColorPolymorphicInputFieldTemplate,
  ConditioningCollectionInputFieldTemplate,
  ConditioningField,
  ConditioningInputFieldTemplate,
  ConditioningPolymorphicInputFieldTemplate,
  ControlCollectionInputFieldTemplate,
  ControlInputFieldTemplate,
  ControlNetModelInputFieldTemplate,
  ControlPolymorphicInputFieldTemplate,
  DenoiseMaskInputFieldTemplate,
  EnumInputFieldTemplate,
  FieldType,
  FloatCollectionInputFieldTemplate,
  FloatInputFieldTemplate,
  FloatPolymorphicInputFieldTemplate,
  IPAdapterCollectionInputFieldTemplate,
  IPAdapterField,
  IPAdapterInputFieldTemplate,
  IPAdapterModelInputFieldTemplate,
  IPAdapterPolymorphicInputFieldTemplate,
  ImageCollectionInputFieldTemplate,
  ImageField,
  ImageInputFieldTemplate,
  ImagePolymorphicInputFieldTemplate,
  InputFieldTemplate,
  InputFieldTemplateBase,
  IntegerCollectionInputFieldTemplate,
  IntegerInputFieldTemplate,
  IntegerPolymorphicInputFieldTemplate,
  InvocationFieldSchema,
  InvocationSchemaObject,
  LatentsCollectionInputFieldTemplate,
  LatentsField,
  LatentsInputFieldTemplate,
  LatentsPolymorphicInputFieldTemplate,
  LoRAModelInputFieldTemplate,
  MainModelInputFieldTemplate,
  MetadataCollectionInputFieldTemplate,
  MetadataInputFieldTemplate,
  MetadataItemCollectionInputFieldTemplate,
  MetadataItemInputFieldTemplate,
  MetadataItemPolymorphicInputFieldTemplate,
  OpenAPIV3_1SchemaOrRef,
  SDXLMainModelInputFieldTemplate,
  SDXLRefinerModelInputFieldTemplate,
  SchedulerInputFieldTemplate,
  StringCollectionInputFieldTemplate,
  StringInputFieldTemplate,
  StringPolymorphicInputFieldTemplate,
  T2IAdapterCollectionInputFieldTemplate,
  T2IAdapterField,
  T2IAdapterInputFieldTemplate,
  T2IAdapterModelInputFieldTemplate,
  T2IAdapterPolymorphicInputFieldTemplate,
  UNetInputFieldTemplate,
  CustomInputFieldTemplate,
  VaeInputFieldTemplate,
  VaeModelInputFieldTemplate,
  isArraySchemaObject,
  isNonArraySchemaObject,
  isRefObject,
  isSchemaObject,
  isFieldType,
  CustomCollectionInputFieldTemplate,
  CustomPolymorphicInputFieldTemplate,
} from '../types/types';

export type BaseFieldProperties = 'name' | 'title' | 'description';

export type BuildInputFieldArg = {
  schemaObject: InvocationFieldSchema;
  baseField: Omit<InputFieldTemplateBase, 'type'>;
};

/**
 * Transforms an invocation output ref object to field type.
 * @param ref The ref string to transform
 * @returns The field type.
 *
 * @example
 * refObjectToFieldType({ "$ref": "#/components/schemas/ImageField" }) --> 'ImageField'
 */
export const refObjectToSchemaName = (refObject: OpenAPIV3_1.ReferenceObject) =>
  refObject.$ref.split('/').slice(-1)[0];

const buildIntegerInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): IntegerInputFieldTemplate => {
  const template: IntegerInputFieldTemplate = {
    ...baseField,
    type: 'integer',
    default: schemaObject.default ?? 0,
  };

  if (schemaObject.multipleOf !== undefined) {
    template.multipleOf = schemaObject.multipleOf;
  }

  if (schemaObject.maximum !== undefined) {
    template.maximum = schemaObject.maximum;
  }

  if (
    schemaObject.exclusiveMaximum !== undefined &&
    isNumber(schemaObject.exclusiveMaximum)
  ) {
    template.exclusiveMaximum = schemaObject.exclusiveMaximum;
  }

  if (schemaObject.minimum !== undefined) {
    template.minimum = schemaObject.minimum;
  }

  if (
    schemaObject.exclusiveMinimum !== undefined &&
    isNumber(schemaObject.exclusiveMinimum)
  ) {
    template.exclusiveMinimum = schemaObject.exclusiveMinimum;
  }

  return template;
};

const buildIntegerPolymorphicInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): IntegerPolymorphicInputFieldTemplate => {
  const template: IntegerPolymorphicInputFieldTemplate = {
    ...baseField,
    type: 'IntegerPolymorphic',
    default: schemaObject.default ?? 0,
  };

  if (schemaObject.multipleOf !== undefined) {
    template.multipleOf = schemaObject.multipleOf;
  }

  if (schemaObject.maximum !== undefined) {
    template.maximum = schemaObject.maximum;
  }

  if (
    schemaObject.exclusiveMaximum !== undefined &&
    isNumber(schemaObject.exclusiveMaximum)
  ) {
    template.exclusiveMaximum = schemaObject.exclusiveMaximum;
  }

  if (schemaObject.minimum !== undefined) {
    template.minimum = schemaObject.minimum;
  }

  if (
    schemaObject.exclusiveMinimum !== undefined &&
    isNumber(schemaObject.exclusiveMinimum)
  ) {
    template.exclusiveMinimum = schemaObject.exclusiveMinimum;
  }

  return template;
};

const buildIntegerCollectionInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): IntegerCollectionInputFieldTemplate => {
  const item_default =
    isNumber(schemaObject.item_default) && isInteger(schemaObject.item_default)
      ? schemaObject.item_default
      : 0;
  const template: IntegerCollectionInputFieldTemplate = {
    ...baseField,
    type: 'IntegerCollection',
    default: schemaObject.default ?? [],
    item_default,
  };

  return template;
};

const buildFloatInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): FloatInputFieldTemplate => {
  const template: FloatInputFieldTemplate = {
    ...baseField,
    type: 'float',
    default: schemaObject.default ?? 0,
  };

  if (schemaObject.multipleOf !== undefined) {
    template.multipleOf = schemaObject.multipleOf;
  }

  if (schemaObject.maximum !== undefined) {
    template.maximum = schemaObject.maximum;
  }

  if (
    schemaObject.exclusiveMaximum !== undefined &&
    isNumber(schemaObject.exclusiveMaximum)
  ) {
    template.exclusiveMaximum = schemaObject.exclusiveMaximum;
  }

  if (schemaObject.minimum !== undefined) {
    template.minimum = schemaObject.minimum;
  }

  if (
    schemaObject.exclusiveMinimum !== undefined &&
    isNumber(schemaObject.exclusiveMinimum)
  ) {
    template.exclusiveMinimum = schemaObject.exclusiveMinimum;
  }

  return template;
};

const buildFloatPolymorphicInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): FloatPolymorphicInputFieldTemplate => {
  const template: FloatPolymorphicInputFieldTemplate = {
    ...baseField,
    type: 'FloatPolymorphic',
    default: schemaObject.default ?? 0,
  };
  if (schemaObject.multipleOf !== undefined) {
    template.multipleOf = schemaObject.multipleOf;
  }

  if (schemaObject.maximum !== undefined) {
    template.maximum = schemaObject.maximum;
  }

  if (
    schemaObject.exclusiveMaximum !== undefined &&
    isNumber(schemaObject.exclusiveMaximum)
  ) {
    template.exclusiveMaximum = schemaObject.exclusiveMaximum;
  }

  if (schemaObject.minimum !== undefined) {
    template.minimum = schemaObject.minimum;
  }

  if (
    schemaObject.exclusiveMinimum !== undefined &&
    isNumber(schemaObject.exclusiveMinimum)
  ) {
    template.exclusiveMinimum = schemaObject.exclusiveMinimum;
  }
  return template;
};

const buildFloatCollectionInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): FloatCollectionInputFieldTemplate => {
  const item_default = isNumber(schemaObject.item_default)
    ? schemaObject.item_default
    : 0;
  const template: FloatCollectionInputFieldTemplate = {
    ...baseField,
    type: 'FloatCollection',
    default: schemaObject.default ?? [],
    item_default,
  };

  return template;
};

const buildStringInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): StringInputFieldTemplate => {
  const template: StringInputFieldTemplate = {
    ...baseField,
    type: 'string',
    default: schemaObject.default ?? '',
  };

  if (schemaObject.minLength !== undefined) {
    template.minLength = schemaObject.minLength;
  }

  if (schemaObject.maxLength !== undefined) {
    template.maxLength = schemaObject.maxLength;
  }

  if (schemaObject.pattern !== undefined) {
    template.pattern = schemaObject.pattern;
  }

  return template;
};

const buildStringPolymorphicInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): StringPolymorphicInputFieldTemplate => {
  const template: StringPolymorphicInputFieldTemplate = {
    ...baseField,
    type: 'StringPolymorphic',
    default: schemaObject.default ?? '',
  };

  if (schemaObject.minLength !== undefined) {
    template.minLength = schemaObject.minLength;
  }

  if (schemaObject.maxLength !== undefined) {
    template.maxLength = schemaObject.maxLength;
  }

  if (schemaObject.pattern !== undefined) {
    template.pattern = schemaObject.pattern;
  }

  return template;
};

const buildStringCollectionInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): StringCollectionInputFieldTemplate => {
  const item_default = isString(schemaObject.item_default)
    ? schemaObject.item_default
    : '';
  const template: StringCollectionInputFieldTemplate = {
    ...baseField,
    type: 'StringCollection',
    default: schemaObject.default ?? [],
    item_default,
  };

  return template;
};

const buildBooleanInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): BooleanInputFieldTemplate => {
  const template: BooleanInputFieldTemplate = {
    ...baseField,
    type: 'boolean',
    default: schemaObject.default ?? false,
  };

  return template;
};

const buildBooleanPolymorphicInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): BooleanPolymorphicInputFieldTemplate => {
  const template: BooleanPolymorphicInputFieldTemplate = {
    ...baseField,
    type: 'BooleanPolymorphic',
    default: schemaObject.default ?? false,
  };

  return template;
};

const buildBooleanCollectionInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): BooleanCollectionInputFieldTemplate => {
  const item_default =
    schemaObject.item_default && isBoolean(schemaObject.item_default)
      ? schemaObject.item_default
      : false;
  const template: BooleanCollectionInputFieldTemplate = {
    ...baseField,
    type: 'BooleanCollection',
    default: schemaObject.default ?? [],
    item_default,
  };

  return template;
};

const buildMainModelInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): MainModelInputFieldTemplate => {
  const template: MainModelInputFieldTemplate = {
    ...baseField,
    type: 'MainModelField',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildSDXLMainModelInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): SDXLMainModelInputFieldTemplate => {
  const template: SDXLMainModelInputFieldTemplate = {
    ...baseField,
    type: 'SDXLMainModelField',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildRefinerModelInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): SDXLRefinerModelInputFieldTemplate => {
  const template: SDXLRefinerModelInputFieldTemplate = {
    ...baseField,
    type: 'SDXLRefinerModelField',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildVaeModelInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): VaeModelInputFieldTemplate => {
  const template: VaeModelInputFieldTemplate = {
    ...baseField,
    type: 'VaeModelField',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildLoRAModelInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): LoRAModelInputFieldTemplate => {
  const template: LoRAModelInputFieldTemplate = {
    ...baseField,
    type: 'LoRAModelField',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildControlNetModelInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): ControlNetModelInputFieldTemplate => {
  const template: ControlNetModelInputFieldTemplate = {
    ...baseField,
    type: 'ControlNetModelField',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildIPAdapterModelInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): IPAdapterModelInputFieldTemplate => {
  const template: IPAdapterModelInputFieldTemplate = {
    ...baseField,
    type: 'IPAdapterModelField',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildT2IAdapterModelInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): T2IAdapterModelInputFieldTemplate => {
  const template: T2IAdapterModelInputFieldTemplate = {
    ...baseField,
    type: 'T2IAdapterModelField',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildBoardInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): BoardInputFieldTemplate => {
  const template: BoardInputFieldTemplate = {
    ...baseField,
    type: 'BoardField',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildImageInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): ImageInputFieldTemplate => {
  const template: ImageInputFieldTemplate = {
    ...baseField,
    type: 'ImageField',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildImagePolymorphicInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): ImagePolymorphicInputFieldTemplate => {
  const template: ImagePolymorphicInputFieldTemplate = {
    ...baseField,
    type: 'ImagePolymorphic',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildImageCollectionInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): ImageCollectionInputFieldTemplate => {
  const template: ImageCollectionInputFieldTemplate = {
    ...baseField,
    type: 'ImageCollection',
    default: schemaObject.default ?? [],
    item_default: (schemaObject.item_default as ImageField) ?? undefined,
  };

  return template;
};

const buildDenoiseMaskInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): DenoiseMaskInputFieldTemplate => {
  const template: DenoiseMaskInputFieldTemplate = {
    ...baseField,
    type: 'DenoiseMaskField',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildLatentsInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): LatentsInputFieldTemplate => {
  const template: LatentsInputFieldTemplate = {
    ...baseField,
    type: 'LatentsField',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildLatentsPolymorphicInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): LatentsPolymorphicInputFieldTemplate => {
  const template: LatentsPolymorphicInputFieldTemplate = {
    ...baseField,
    type: 'LatentsPolymorphic',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildLatentsCollectionInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): LatentsCollectionInputFieldTemplate => {
  const template: LatentsCollectionInputFieldTemplate = {
    ...baseField,
    type: 'LatentsCollection',
    default: schemaObject.default ?? [],
    item_default: (schemaObject.item_default as LatentsField) ?? undefined,
  };

  return template;
};

const buildConditioningInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): ConditioningInputFieldTemplate => {
  const template: ConditioningInputFieldTemplate = {
    ...baseField,
    type: 'ConditioningField',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildConditioningPolymorphicInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): ConditioningPolymorphicInputFieldTemplate => {
  const template: ConditioningPolymorphicInputFieldTemplate = {
    ...baseField,
    type: 'ConditioningPolymorphic',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildConditioningCollectionInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): ConditioningCollectionInputFieldTemplate => {
  const template: ConditioningCollectionInputFieldTemplate = {
    ...baseField,
    type: 'ConditioningCollection',
    default: schemaObject.default ?? [],
    item_default: (schemaObject.item_default as ConditioningField) ?? undefined,
  };

  return template;
};

const buildUNetInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): UNetInputFieldTemplate => {
  const template: UNetInputFieldTemplate = {
    ...baseField,
    type: 'UNetField',

    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildClipInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): ClipInputFieldTemplate => {
  const template: ClipInputFieldTemplate = {
    ...baseField,
    type: 'ClipField',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildVaeInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): VaeInputFieldTemplate => {
  const template: VaeInputFieldTemplate = {
    ...baseField,
    type: 'VaeField',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildControlInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): ControlInputFieldTemplate => {
  const template: ControlInputFieldTemplate = {
    ...baseField,
    type: 'ControlField',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildControlPolymorphicInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): ControlPolymorphicInputFieldTemplate => {
  const template: ControlPolymorphicInputFieldTemplate = {
    ...baseField,
    type: 'ControlPolymorphic',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildControlCollectionInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): ControlCollectionInputFieldTemplate => {
  const template: ControlCollectionInputFieldTemplate = {
    ...baseField,
    type: 'ControlCollection',
    default: schemaObject.default ?? [],
    item_default: (schemaObject.item_default as ControlField) ?? undefined,
  };

  return template;
};

const buildIPAdapterInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): IPAdapterInputFieldTemplate => {
  const template: IPAdapterInputFieldTemplate = {
    ...baseField,
    type: 'IPAdapterField',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildIPAdapterPolymorphicInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): IPAdapterPolymorphicInputFieldTemplate => {
  const template: IPAdapterPolymorphicInputFieldTemplate = {
    ...baseField,
    type: 'IPAdapterPolymorphic',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildIPAdapterCollectionInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): IPAdapterCollectionInputFieldTemplate => {
  const template: IPAdapterCollectionInputFieldTemplate = {
    ...baseField,
    type: 'IPAdapterCollection',
    default: schemaObject.default ?? [],
    item_default: (schemaObject.item_default as IPAdapterField) ?? undefined,
  };

  return template;
};

const buildT2IAdapterInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): T2IAdapterInputFieldTemplate => {
  const template: T2IAdapterInputFieldTemplate = {
    ...baseField,
    type: 'T2IAdapterField',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildT2IAdapterPolymorphicInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): T2IAdapterPolymorphicInputFieldTemplate => {
  const template: T2IAdapterPolymorphicInputFieldTemplate = {
    ...baseField,
    type: 'T2IAdapterPolymorphic',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildT2IAdapterCollectionInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): T2IAdapterCollectionInputFieldTemplate => {
  const template: T2IAdapterCollectionInputFieldTemplate = {
    ...baseField,
    type: 'T2IAdapterCollection',
    default: schemaObject.default ?? [],
    item_default: (schemaObject.item_default as T2IAdapterField) ?? undefined,
  };

  return template;
};

const buildEnumInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): EnumInputFieldTemplate => {
  const options = schemaObject.enum ?? [];
  const template: EnumInputFieldTemplate = {
    ...baseField,
    type: 'enum',
    options,
    ui_choice_labels: schemaObject.ui_choice_labels,
    default: schemaObject.default ?? options[0],
  };

  return template;
};

const buildCollectionInputFieldTemplate = ({
  baseField,
}: BuildInputFieldArg): CollectionInputFieldTemplate => {
  const template: CollectionInputFieldTemplate = {
    ...baseField,
    type: 'Collection',
    default: [],
  };

  return template;
};

const buildCollectionItemInputFieldTemplate = ({
  baseField,
}: BuildInputFieldArg): CollectionItemInputFieldTemplate => {
  const template: CollectionItemInputFieldTemplate = {
    ...baseField,
    type: 'CollectionItem',
    default: undefined,
  };

  return template;
};

const buildAnyInputFieldTemplate = ({
  baseField,
}: BuildInputFieldArg): AnyInputFieldTemplate => {
  const template: AnyInputFieldTemplate = {
    ...baseField,
    type: 'Any',
    default: undefined,
  };

  return template;
};

const buildMetadataItemInputFieldTemplate = ({
  baseField,
}: BuildInputFieldArg): MetadataItemInputFieldTemplate => {
  const template: MetadataItemInputFieldTemplate = {
    ...baseField,
    type: 'MetadataItemField',
    default: undefined,
  };

  return template;
};

const buildMetadataItemCollectionInputFieldTemplate = ({
  baseField,
}: BuildInputFieldArg): MetadataItemCollectionInputFieldTemplate => {
  const template: MetadataItemCollectionInputFieldTemplate = {
    ...baseField,
    type: 'MetadataItemCollection',
    default: undefined,
  };

  return template;
};

const buildMetadataItemPolymorphicInputFieldTemplate = ({
  baseField,
}: BuildInputFieldArg): MetadataItemPolymorphicInputFieldTemplate => {
  const template: MetadataItemPolymorphicInputFieldTemplate = {
    ...baseField,
    type: 'MetadataItemPolymorphic',
    default: undefined,
  };

  return template;
};

const buildMetadataDictInputFieldTemplate = ({
  baseField,
}: BuildInputFieldArg): MetadataInputFieldTemplate => {
  const template: MetadataInputFieldTemplate = {
    ...baseField,
    type: 'MetadataField',
    default: undefined,
  };

  return template;
};

const buildMetadataCollectionInputFieldTemplate = ({
  baseField,
}: BuildInputFieldArg): MetadataCollectionInputFieldTemplate => {
  const template: MetadataCollectionInputFieldTemplate = {
    ...baseField,
    type: 'MetadataCollection',
    default: undefined,
  };

  return template;
};

const buildColorInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): ColorInputFieldTemplate => {
  const template: ColorInputFieldTemplate = {
    ...baseField,
    type: 'ColorField',
    default: schemaObject.default ?? { r: 127, g: 127, b: 127, a: 255 },
  };

  return template;
};

const buildColorPolymorphicInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): ColorPolymorphicInputFieldTemplate => {
  const template: ColorPolymorphicInputFieldTemplate = {
    ...baseField,
    type: 'ColorPolymorphic',
    default: schemaObject.default ?? { r: 127, g: 127, b: 127, a: 255 },
  };

  return template;
};

const buildColorCollectionInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): ColorCollectionInputFieldTemplate => {
  const template: ColorCollectionInputFieldTemplate = {
    ...baseField,
    type: 'ColorCollection',
    default: schemaObject.default ?? [],
  };

  return template;
};

const buildSchedulerInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): SchedulerInputFieldTemplate => {
  const template: SchedulerInputFieldTemplate = {
    ...baseField,
    type: 'Scheduler',
    default: schemaObject.default ?? 'euler',
  };

  return template;
};

const buildCustomCollectionInputFieldTemplate = ({
  baseField,
}: BuildInputFieldArg): CustomCollectionInputFieldTemplate => {
  const template: CustomCollectionInputFieldTemplate = {
    ...baseField,
    type: 'CustomCollection',
    default: [],
  };

  return template;
};

const buildCustomPolymorphicInputFieldTemplate = ({
  baseField,
}: BuildInputFieldArg): CustomPolymorphicInputFieldTemplate => {
  const template: CustomPolymorphicInputFieldTemplate = {
    ...baseField,
    type: 'CustomPolymorphic',
    default: undefined,
  };

  return template;
};

const buildCustomInputFieldTemplate = ({
  baseField,
}: BuildInputFieldArg): CustomInputFieldTemplate => {
  const template: CustomInputFieldTemplate = {
    ...baseField,
    type: 'Custom',
    default: undefined,
  };

  return template;
};

export const getFieldType = (
  schemaObject: OpenAPIV3_1SchemaOrRef
): { type: string; originalType: string } | undefined => {
  if (isSchemaObject(schemaObject)) {
    if (!schemaObject.type) {
      // if schemaObject has no type, then it should have one of allOf, anyOf, oneOf

      if (schemaObject.allOf) {
        const allOf = schemaObject.allOf;
        if (allOf && allOf[0] && isRefObject(allOf[0])) {
          // This is a single ref type
          const originalType = refObjectToSchemaName(allOf[0]);

          if (!originalType) {
            // something has gone terribly awry
            return;
          }
          return {
            type: isFieldType(originalType) ? originalType : 'Custom',
            originalType,
          };
        }
      } else if (schemaObject.anyOf) {
        // ignore null types
        const anyOf = schemaObject.anyOf.filter((i) => {
          if (isSchemaObject(i)) {
            if (i.type === 'null') {
              return false;
            }
          }
          return true;
        });
        if (anyOf.length === 1) {
          // This is a single ref type
          if (isRefObject(anyOf[0])) {
            const originalType = refObjectToSchemaName(anyOf[0]);
            if (!originalType) {
              return;
            }

            return {
              type: isFieldType(originalType) ? originalType : 'Custom',
              originalType,
            };
          } else if (isSchemaObject(anyOf[0])) {
            return getFieldType(anyOf[0]);
          }
        }
        /**
         * Handle Polymorphic inputs, eg string | string[]. In OpenAPI, this is:
         * - an `anyOf` with two items
         * - one is an `ArraySchemaObject` with a single `SchemaObject or ReferenceObject` of type T in its `items`
         * - the other is a `SchemaObject` or `ReferenceObject` of type T
         *
         * Any other cases we ignore.
         */

        let firstType: string | undefined;
        let secondType: string | undefined;

        if (isArraySchemaObject(anyOf[0])) {
          // first is array, second is not
          const first = anyOf[0].items;
          const second = anyOf[1];
          if (isRefObject(first) && isRefObject(second)) {
            firstType = refObjectToSchemaName(first);
            secondType = refObjectToSchemaName(second);
          } else if (
            isNonArraySchemaObject(first) &&
            isNonArraySchemaObject(second)
          ) {
            firstType = first.type;
            secondType = second.type;
          }
        } else if (isArraySchemaObject(anyOf[1])) {
          // first is not array, second is
          const first = anyOf[0];
          const second = anyOf[1].items;
          if (isRefObject(first) && isRefObject(second)) {
            firstType = refObjectToSchemaName(first);
            secondType = refObjectToSchemaName(second);
          } else if (
            isNonArraySchemaObject(first) &&
            isNonArraySchemaObject(second)
          ) {
            firstType = first.type;
            secondType = second.type;
          }
        }
        if (firstType === secondType) {
          if (isPolymorphicItemType(firstType)) {
            // Known polymorphic field type
            const originalType = SINGLE_TO_POLYMORPHIC_MAP[firstType];
            if (!originalType) {
              return;
            }
            return { type: originalType, originalType };
          }

          // else custom polymorphic
          return {
            type: 'CustomPolymorphic',
            originalType: `${firstType}Polymorphic`,
          };
        }
      }
    } else if (schemaObject.enum) {
      return { type: 'enum', originalType: 'enum' };
    } else if (schemaObject.type) {
      if (schemaObject.type === 'number') {
        // floats are "number" in OpenAPI, while ints are "integer" - we need to distinguish them
        return { type: 'float', originalType: 'float' };
      } else if (schemaObject.type === 'array') {
        const itemType = isSchemaObject(schemaObject.items)
          ? schemaObject.items.type
          : refObjectToSchemaName(schemaObject.items);

        if (isArray(itemType)) {
          // This is a nested array, which we don't support
          return;
        }

        if (isCollectionItemType(itemType)) {
          // known collection field type
          const originalType = COLLECTION_MAP[itemType];
          if (!originalType) {
            return;
          }
          return { type: originalType, originalType };
        }

        return {
          type: 'CustomCollection',
          originalType: `${itemType}Collection`,
        };
      } else if (
        !isArray(schemaObject.type) &&
        schemaObject.type !== 'null' && // 'null' is not valid
        schemaObject.type !== 'object' // 'object' is not valid
      ) {
        const originalType = schemaObject.type;
        return { type: originalType, originalType };
      }
      // else ignore
      return;
    }
  } else if (isRefObject(schemaObject)) {
    const originalType = refObjectToSchemaName(schemaObject);
    if (!originalType) {
      return;
    }

    return {
      type: isFieldType(originalType) ? originalType : 'Custom',
      originalType,
    };
  }
  return;
};

const TEMPLATE_BUILDER_MAP: {
  [key in FieldType]?: (arg: BuildInputFieldArg) => InputFieldTemplate;
} = {
  BoardField: buildBoardInputFieldTemplate,
  Any: buildAnyInputFieldTemplate,
  boolean: buildBooleanInputFieldTemplate,
  BooleanCollection: buildBooleanCollectionInputFieldTemplate,
  BooleanPolymorphic: buildBooleanPolymorphicInputFieldTemplate,
  ClipField: buildClipInputFieldTemplate,
  Collection: buildCollectionInputFieldTemplate,
  CollectionItem: buildCollectionItemInputFieldTemplate,
  ColorCollection: buildColorCollectionInputFieldTemplate,
  ColorField: buildColorInputFieldTemplate,
  ColorPolymorphic: buildColorPolymorphicInputFieldTemplate,
  ConditioningCollection: buildConditioningCollectionInputFieldTemplate,
  ConditioningField: buildConditioningInputFieldTemplate,
  ConditioningPolymorphic: buildConditioningPolymorphicInputFieldTemplate,
  ControlCollection: buildControlCollectionInputFieldTemplate,
  ControlField: buildControlInputFieldTemplate,
  ControlNetModelField: buildControlNetModelInputFieldTemplate,
  ControlPolymorphic: buildControlPolymorphicInputFieldTemplate,
  DenoiseMaskField: buildDenoiseMaskInputFieldTemplate,
  enum: buildEnumInputFieldTemplate,
  float: buildFloatInputFieldTemplate,
  FloatCollection: buildFloatCollectionInputFieldTemplate,
  FloatPolymorphic: buildFloatPolymorphicInputFieldTemplate,
  ImageCollection: buildImageCollectionInputFieldTemplate,
  ImageField: buildImageInputFieldTemplate,
  ImagePolymorphic: buildImagePolymorphicInputFieldTemplate,
  integer: buildIntegerInputFieldTemplate,
  IntegerCollection: buildIntegerCollectionInputFieldTemplate,
  IntegerPolymorphic: buildIntegerPolymorphicInputFieldTemplate,
  IPAdapterCollection: buildIPAdapterCollectionInputFieldTemplate,
  IPAdapterField: buildIPAdapterInputFieldTemplate,
  IPAdapterModelField: buildIPAdapterModelInputFieldTemplate,
  IPAdapterPolymorphic: buildIPAdapterPolymorphicInputFieldTemplate,
  LatentsCollection: buildLatentsCollectionInputFieldTemplate,
  LatentsField: buildLatentsInputFieldTemplate,
  LatentsPolymorphic: buildLatentsPolymorphicInputFieldTemplate,
  LoRAModelField: buildLoRAModelInputFieldTemplate,
  MetadataItemField: buildMetadataItemInputFieldTemplate,
  MetadataItemCollection: buildMetadataItemCollectionInputFieldTemplate,
  MetadataItemPolymorphic: buildMetadataItemPolymorphicInputFieldTemplate,
  MetadataField: buildMetadataDictInputFieldTemplate,
  MetadataCollection: buildMetadataCollectionInputFieldTemplate,
  MainModelField: buildMainModelInputFieldTemplate,
  Scheduler: buildSchedulerInputFieldTemplate,
  SDXLMainModelField: buildSDXLMainModelInputFieldTemplate,
  SDXLRefinerModelField: buildRefinerModelInputFieldTemplate,
  string: buildStringInputFieldTemplate,
  StringCollection: buildStringCollectionInputFieldTemplate,
  StringPolymorphic: buildStringPolymorphicInputFieldTemplate,
  T2IAdapterCollection: buildT2IAdapterCollectionInputFieldTemplate,
  T2IAdapterField: buildT2IAdapterInputFieldTemplate,
  T2IAdapterModelField: buildT2IAdapterModelInputFieldTemplate,
  T2IAdapterPolymorphic: buildT2IAdapterPolymorphicInputFieldTemplate,
  UNetField: buildUNetInputFieldTemplate,
  VaeField: buildVaeInputFieldTemplate,
  VaeModelField: buildVaeModelInputFieldTemplate,
  Custom: buildCustomInputFieldTemplate,
  CustomCollection: buildCustomCollectionInputFieldTemplate,
  CustomPolymorphic: buildCustomPolymorphicInputFieldTemplate,
};

/**
 * Builds an input field from an invocation schema property.
 * @param fieldSchema The schema object
 * @returns An input field
 */
export const buildInputFieldTemplate = (
  nodeSchema: InvocationSchemaObject,
  fieldSchema: InvocationFieldSchema,
  name: string,
  fieldType: FieldType,
  originalType: string
) => {
  const {
    input,
    ui_hidden,
    ui_component,
    ui_type,
    ui_order,
    ui_choice_labels,
    item_default,
  } = fieldSchema;

  const extra = {
    // TODO: Can we support polymorphic inputs in the UI?
    input: POLYMORPHIC_TYPES.includes(fieldType) ? 'connection' : input,
    ui_hidden,
    ui_component,
    ui_type,
    required: nodeSchema.required?.includes(name) ?? false,
    ui_order,
    ui_choice_labels,
    item_default,
    originalType,
  };

  const baseField = {
    name,
    title: fieldSchema.title ?? (name ? startCase(name) : ''),
    description: fieldSchema.description ?? '',
    fieldKind: 'input' as const,
    ...extra,
  };

  const builder = TEMPLATE_BUILDER_MAP[fieldType];

  if (!builder) {
    return;
  }

  return builder({
    schemaObject: fieldSchema,
    baseField,
  });
};
