import { isNil, trim } from 'lodash-es';
import MersenneTwister from 'mtwist';
import { assert } from 'tsafe';
import { z } from 'zod';

import { zBoardField, zColorField, zImageField, zModelIdentifierField, zSchedulerField } from './common';

/**
 * zod schemas & inferred types for fields.
 *
 * These schemas and types are only required for stateful field - fields that have UI components
 * and allow the user to directly provide values.
 *
 * This includes primitive values (numbers, strings, booleans), models, scheduler, etc.
 *
 * If a field type does not have a UI component, then it does not need to be included here, because
 * we never store its value. Such field types will be handled via the "StatelessField" logic.
 *
 * Fields require:
 * - z<TypeName>FieldType - zod schema for the field type
 * - z<TypeName>FieldValue - zod schema for the field value
 * - z<TypeName>FieldInputInstance - zod schema for the field's input instance
 * - z<TypeName>FieldOutputInstance - zod schema for the field's output instance
 * - z<TypeName>FieldInputTemplate - zod schema for the field's input template
 * - z<TypeName>FieldOutputTemplate - zod schema for the field's output template
 * - inferred types for each schema
 * - type guards for InputInstance and InputTemplate
 *
 * These then must be added to the unions at the bottom of this file.
 */

/** */

// #region Base schemas & misc
const zFieldInput = z.enum(['connection', 'direct', 'any']);
const zFieldUIComponent = z.enum(['none', 'textarea', 'slider']);
const zFieldInputInstanceBase = z.object({
  name: z.string().trim().min(1),
  label: z.string().catch(''),
  description: z.string().catch(''),
});
const zFieldTemplateBase = z.object({
  name: z.string().min(1),
  title: z.string().min(1),
  description: z.string().catch(''),
  ui_hidden: z.boolean(),
  ui_type: z.string().nullish(),
  ui_order: z.number().int().nullish(),
});
const zFieldInputTemplateBase = zFieldTemplateBase.extend({
  fieldKind: z.literal('input'),
  input: zFieldInput,
  required: z.boolean(),
  ui_component: zFieldUIComponent.nullish(),
  ui_choice_labels: z.record(z.string()).nullish(),
});
const zFieldOutputTemplateBase = zFieldTemplateBase.extend({
  fieldKind: z.literal('output'),
});

const SINGLE = 'SINGLE' as const;
const COLLECTION = 'COLLECTION' as const;
const SINGLE_OR_COLLECTION = 'SINGLE_OR_COLLECTION' as const;
const zCardinality = z.enum([SINGLE, COLLECTION, SINGLE_OR_COLLECTION]);

const zFieldTypeBase = z.object({
  cardinality: zCardinality,
  batch: z.boolean(),
});

export const zFieldIdentifier = z.object({
  nodeId: z.string().trim().min(1),
  fieldName: z.string().trim().min(1),
});
export type FieldIdentifier = z.infer<typeof zFieldIdentifier>;
// #endregion

// #region Field Types
const zStatelessFieldType = zFieldTypeBase.extend({
  name: z.string().min(1), // stateless --> we accept the field's name as the type
});
const zIntegerFieldType = zFieldTypeBase.extend({
  name: z.literal('IntegerField'),
  originalType: zStatelessFieldType.optional(),
});
const zIntegerCollectionFieldType = zFieldTypeBase.extend({
  name: z.literal('IntegerField'),
  cardinality: z.literal(COLLECTION),
  originalType: zStatelessFieldType.optional(),
});
export const isIntegerCollectionFieldType = (
  fieldType: FieldType
): fieldType is z.infer<typeof zIntegerCollectionFieldType> =>
  fieldType.name === 'IntegerField' && fieldType.cardinality === COLLECTION;

const zFloatFieldType = zFieldTypeBase.extend({
  name: z.literal('FloatField'),
  originalType: zStatelessFieldType.optional(),
});
const zFloatCollectionFieldType = zFieldTypeBase.extend({
  name: z.literal('FloatField'),
  cardinality: z.literal(COLLECTION),
  originalType: zStatelessFieldType.optional(),
});
export const isFloatCollectionFieldType = (
  fieldType: FieldType
): fieldType is z.infer<typeof zFloatCollectionFieldType> =>
  fieldType.name === 'FloatField' && fieldType.cardinality === COLLECTION;

const zStringFieldType = zFieldTypeBase.extend({
  name: z.literal('StringField'),
  originalType: zStatelessFieldType.optional(),
});
const zStringCollectionFieldType = zFieldTypeBase.extend({
  name: z.literal('StringField'),
  cardinality: z.literal(COLLECTION),
  originalType: zStatelessFieldType.optional(),
});
export const isStringCollectionFieldType = (
  fieldType: FieldType
): fieldType is z.infer<typeof zStringCollectionFieldType> =>
  fieldType.name === 'StringField' && fieldType.cardinality === COLLECTION;

const zBooleanFieldType = zFieldTypeBase.extend({
  name: z.literal('BooleanField'),
  originalType: zStatelessFieldType.optional(),
});
const zEnumFieldType = zFieldTypeBase.extend({
  name: z.literal('EnumField'),
  originalType: zStatelessFieldType.optional(),
});
const zImageFieldType = zFieldTypeBase.extend({
  name: z.literal('ImageField'),
  originalType: zStatelessFieldType.optional(),
});
const zImageCollectionFieldType = zFieldTypeBase.extend({
  name: z.literal('ImageField'),
  cardinality: z.literal(COLLECTION),
  originalType: zStatelessFieldType.optional(),
});
export const isImageCollectionFieldType = (
  fieldType: FieldType
): fieldType is z.infer<typeof zImageCollectionFieldType> =>
  fieldType.name === 'ImageField' && fieldType.cardinality === COLLECTION;

const zBoardFieldType = zFieldTypeBase.extend({
  name: z.literal('BoardField'),
  originalType: zStatelessFieldType.optional(),
});
const zColorFieldType = zFieldTypeBase.extend({
  name: z.literal('ColorField'),
  originalType: zStatelessFieldType.optional(),
});
const zMainModelFieldType = zFieldTypeBase.extend({
  name: z.literal('MainModelField'),
  originalType: zStatelessFieldType.optional(),
});
const zModelIdentifierFieldType = zFieldTypeBase.extend({
  name: z.literal('ModelIdentifierField'),
  originalType: zStatelessFieldType.optional(),
});
const zSDXLMainModelFieldType = zFieldTypeBase.extend({
  name: z.literal('SDXLMainModelField'),
  originalType: zStatelessFieldType.optional(),
});
const zSD3MainModelFieldType = zFieldTypeBase.extend({
  name: z.literal('SD3MainModelField'),
  originalType: zStatelessFieldType.optional(),
});
const zFluxMainModelFieldType = zFieldTypeBase.extend({
  name: z.literal('FluxMainModelField'),
  originalType: zStatelessFieldType.optional(),
});
const zSDXLRefinerModelFieldType = zFieldTypeBase.extend({
  name: z.literal('SDXLRefinerModelField'),
  originalType: zStatelessFieldType.optional(),
});
const zVAEModelFieldType = zFieldTypeBase.extend({
  name: z.literal('VAEModelField'),
  originalType: zStatelessFieldType.optional(),
});
const zLoRAModelFieldType = zFieldTypeBase.extend({
  name: z.literal('LoRAModelField'),
  originalType: zStatelessFieldType.optional(),
});
const zControlNetModelFieldType = zFieldTypeBase.extend({
  name: z.literal('ControlNetModelField'),
  originalType: zStatelessFieldType.optional(),
});
const zIPAdapterModelFieldType = zFieldTypeBase.extend({
  name: z.literal('IPAdapterModelField'),
  originalType: zStatelessFieldType.optional(),
});
const zT2IAdapterModelFieldType = zFieldTypeBase.extend({
  name: z.literal('T2IAdapterModelField'),
  originalType: zStatelessFieldType.optional(),
});
const zSpandrelImageToImageModelFieldType = zFieldTypeBase.extend({
  name: z.literal('SpandrelImageToImageModelField'),
  originalType: zStatelessFieldType.optional(),
});
const zT5EncoderModelFieldType = zFieldTypeBase.extend({
  name: z.literal('T5EncoderModelField'),
  originalType: zStatelessFieldType.optional(),
});
const zCLIPEmbedModelFieldType = zFieldTypeBase.extend({
  name: z.literal('CLIPEmbedModelField'),
  originalType: zStatelessFieldType.optional(),
});
const zCLIPLEmbedModelFieldType = zFieldTypeBase.extend({
  name: z.literal('CLIPLEmbedModelField'),
  originalType: zStatelessFieldType.optional(),
});
const zCLIPGEmbedModelFieldType = zFieldTypeBase.extend({
  name: z.literal('CLIPGEmbedModelField'),
  originalType: zStatelessFieldType.optional(),
});
const zControlLoRAModelFieldType = zFieldTypeBase.extend({
  name: z.literal('ControlLoRAModelField'),
  originalType: zStatelessFieldType.optional(),
});
const zFluxVAEModelFieldType = zFieldTypeBase.extend({
  name: z.literal('FluxVAEModelField'),
  originalType: zStatelessFieldType.optional(),
});
const zSchedulerFieldType = zFieldTypeBase.extend({
  name: z.literal('SchedulerField'),
  originalType: zStatelessFieldType.optional(),
});
const zFloatGeneratorFieldType = zFieldTypeBase.extend({
  name: z.literal('FloatGeneratorField'),
  originalType: zStatelessFieldType.optional(),
});
const zIntegerGeneratorFieldType = zFieldTypeBase.extend({
  name: z.literal('IntegerGeneratorField'),
  originalType: zStatelessFieldType.optional(),
});
const zStringGeneratorFieldType = zFieldTypeBase.extend({
  name: z.literal('StringGeneratorField'),
  originalType: zStatelessFieldType.optional(),
});
const zStatefulFieldType = z.union([
  zIntegerFieldType,
  zFloatFieldType,
  zStringFieldType,
  zBooleanFieldType,
  zEnumFieldType,
  zImageFieldType,
  zBoardFieldType,
  zModelIdentifierFieldType,
  zMainModelFieldType,
  zSDXLMainModelFieldType,
  zSD3MainModelFieldType,
  zFluxMainModelFieldType,
  zSDXLRefinerModelFieldType,
  zVAEModelFieldType,
  zLoRAModelFieldType,
  zControlNetModelFieldType,
  zIPAdapterModelFieldType,
  zT2IAdapterModelFieldType,
  zSpandrelImageToImageModelFieldType,
  zT5EncoderModelFieldType,
  zCLIPEmbedModelFieldType,
  zCLIPLEmbedModelFieldType,
  zCLIPGEmbedModelFieldType,
  zControlLoRAModelFieldType,
  zFluxVAEModelFieldType,
  zColorFieldType,
  zSchedulerFieldType,
  zFloatGeneratorFieldType,
  zIntegerGeneratorFieldType,
  zStringGeneratorFieldType,
]);
export type StatefulFieldType = z.infer<typeof zStatefulFieldType>;
const statefulFieldTypeNames = zStatefulFieldType.options.map((o) => o.shape.name.value);
export const isStatefulFieldType = (fieldType: FieldType): fieldType is StatefulFieldType =>
  (statefulFieldTypeNames as string[]).includes(fieldType.name);
const zFieldType = z.union([zStatefulFieldType, zStatelessFieldType]);
export type FieldType = z.infer<typeof zFieldType>;

const modelFieldTypeNames = [
  // Stateful model fields
  zModelIdentifierFieldType.shape.name.value,
  zMainModelFieldType.shape.name.value,
  zSDXLMainModelFieldType.shape.name.value,
  zSD3MainModelFieldType.shape.name.value,
  zFluxMainModelFieldType.shape.name.value,
  zSDXLRefinerModelFieldType.shape.name.value,
  zVAEModelFieldType.shape.name.value,
  zLoRAModelFieldType.shape.name.value,
  zControlNetModelFieldType.shape.name.value,
  zIPAdapterModelFieldType.shape.name.value,
  zT2IAdapterModelFieldType.shape.name.value,
  zSpandrelImageToImageModelFieldType.shape.name.value,
  zT5EncoderModelFieldType.shape.name.value,
  zCLIPEmbedModelFieldType.shape.name.value,
  zCLIPLEmbedModelFieldType.shape.name.value,
  zCLIPGEmbedModelFieldType.shape.name.value,
  zControlLoRAModelFieldType.shape.name.value,
  zFluxVAEModelFieldType.shape.name.value,
  // Stateless model fields
  'UNetField',
  'VAEField',
  'CLIPField',
  'T5EncoderField',
  'TransformerField',
  'ControlLoRAField',
];
export const isModelFieldType = (fieldType: FieldType) => {
  return (modelFieldTypeNames as string[]).includes(fieldType.name);
};

export const isSingle = (fieldType: FieldType): boolean => fieldType.cardinality === zCardinality.enum.SINGLE;
export const isCollection = (fieldType: FieldType): boolean => fieldType.cardinality === zCardinality.enum.COLLECTION;
export const isSingleOrCollection = (fieldType: FieldType): boolean =>
  fieldType.cardinality === zCardinality.enum.SINGLE_OR_COLLECTION;
// #endregion

const buildInstanceTypeGuard = <T extends z.ZodTypeAny>(schema: T) => {
  return (val: unknown): val is z.infer<T> => schema.safeParse(val).success;
};

const buildTemplateTypeGuard =
  <T extends FieldInputTemplate>(name: string, cardinality?: 'SINGLE' | 'COLLECTION' | 'SINGLE_OR_COLLECTION') =>
  (template: FieldInputTemplate): template is T => {
    if (template.type.name !== name) {
      return false;
    }
    if (cardinality) {
      return template.type.cardinality === cardinality;
    }
    return true;
  };

// #region IntegerField

export const zIntegerFieldValue = z.number().int();
const zIntegerFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zIntegerFieldValue,
});
const zIntegerFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zIntegerFieldType,
  originalType: zFieldType.optional(),
  default: zIntegerFieldValue,
  multipleOf: z.number().int().optional(),
  maximum: z.number().int().optional(),
  exclusiveMaximum: z.number().int().optional(),
  minimum: z.number().int().optional(),
  exclusiveMinimum: z.number().int().optional(),
});
const zIntegerFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zIntegerFieldType,
});
export type IntegerFieldValue = z.infer<typeof zIntegerFieldValue>;
export type IntegerFieldInputInstance = z.infer<typeof zIntegerFieldInputInstance>;
export type IntegerFieldInputTemplate = z.infer<typeof zIntegerFieldInputTemplate>;
export const isIntegerFieldInputInstance = buildInstanceTypeGuard(zIntegerFieldInputInstance);
export const isIntegerFieldInputTemplate = buildTemplateTypeGuard<IntegerFieldInputTemplate>('IntegerField', 'SINGLE');
// #endregion

// #region IntegerField Collection
export const zIntegerFieldCollectionValue = z.array(zIntegerFieldValue).optional();
const zIntegerFieldCollectionInputInstance = zFieldInputInstanceBase.extend({
  value: zIntegerFieldCollectionValue,
});
const zIntegerFieldCollectionInputTemplate = zFieldInputTemplateBase
  .extend({
    type: zIntegerCollectionFieldType,
    originalType: zFieldType.optional(),
    default: zIntegerFieldCollectionValue,
    maxItems: z.number().int().gte(0).optional(),
    minItems: z.number().int().gte(0).optional(),
    multipleOf: z.number().int().optional(),
    maximum: z.number().int().optional(),
    exclusiveMaximum: z.number().int().optional(),
    minimum: z.number().int().optional(),
    exclusiveMinimum: z.number().int().optional(),
  })
  .refine(
    (val) => {
      if (val.maxItems !== undefined && val.minItems !== undefined) {
        return val.maxItems >= val.minItems;
      }
      return true;
    },
    { message: 'maxItems must be greater than or equal to minItems' }
  );

const zIntegerFieldCollectionOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zIntegerCollectionFieldType,
});
export type IntegerFieldCollectionValue = z.infer<typeof zIntegerFieldCollectionValue>;
export type IntegerFieldCollectionInputInstance = z.infer<typeof zIntegerFieldCollectionInputInstance>;
export type IntegerFieldCollectionInputTemplate = z.infer<typeof zIntegerFieldCollectionInputTemplate>;
export const isIntegerFieldCollectionInputInstance = buildInstanceTypeGuard(zIntegerFieldCollectionInputInstance);
export const isIntegerFieldCollectionInputTemplate = buildTemplateTypeGuard<IntegerFieldCollectionInputTemplate>(
  'IntegerField',
  'COLLECTION'
);
// #endregion

// #region FloatField
export const zFloatFieldValue = z.number();
const zFloatFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zFloatFieldValue,
});
const zFloatFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zFloatFieldType,
  originalType: zFieldType.optional(),
  default: zFloatFieldValue,
  multipleOf: z.number().optional(),
  maximum: z.number().optional(),
  exclusiveMaximum: z.number().optional(),
  minimum: z.number().optional(),
  exclusiveMinimum: z.number().optional(),
});
const zFloatFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zFloatFieldType,
});
export type FloatFieldValue = z.infer<typeof zFloatFieldValue>;
export type FloatFieldInputInstance = z.infer<typeof zFloatFieldInputInstance>;
export type FloatFieldInputTemplate = z.infer<typeof zFloatFieldInputTemplate>;
export const isFloatFieldInputInstance = buildInstanceTypeGuard(zFloatFieldInputInstance);
export const isFloatFieldInputTemplate = buildTemplateTypeGuard<FloatFieldInputTemplate>('FloatField', 'SINGLE');
// #endregion

// #region FloatField Collection
export const zFloatFieldCollectionValue = z.array(zFloatFieldValue).optional();
const zFloatFieldCollectionInputInstance = zFieldInputInstanceBase.extend({
  value: zFloatFieldCollectionValue,
});
const zFloatFieldCollectionInputTemplate = zFieldInputTemplateBase
  .extend({
    type: zFloatCollectionFieldType,
    originalType: zFieldType.optional(),
    default: zFloatFieldCollectionValue,
    maxItems: z.number().int().gte(0).optional(),
    minItems: z.number().int().gte(0).optional(),
    multipleOf: z.number().int().optional(),
    maximum: z.number().optional(),
    exclusiveMaximum: z.number().optional(),
    minimum: z.number().optional(),
    exclusiveMinimum: z.number().optional(),
  })
  .refine(
    (val) => {
      if (val.maxItems !== undefined && val.minItems !== undefined) {
        return val.maxItems >= val.minItems;
      }
      return true;
    },
    { message: 'maxItems must be greater than or equal to minItems' }
  );
const zFloatFieldCollectionOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zFloatCollectionFieldType,
});
export type FloatFieldCollectionValue = z.infer<typeof zFloatFieldCollectionValue>;
export type FloatFieldCollectionInputInstance = z.infer<typeof zFloatFieldCollectionInputInstance>;
export type FloatFieldCollectionInputTemplate = z.infer<typeof zFloatFieldCollectionInputTemplate>;
export const isFloatFieldCollectionInputInstance = buildInstanceTypeGuard(zFloatFieldCollectionInputInstance);
export const isFloatFieldCollectionInputTemplate = buildTemplateTypeGuard<FloatFieldCollectionInputTemplate>(
  'FloatField',
  'COLLECTION'
);
// #endregion

// #region StringField
export const zStringFieldValue = z.string();
const zStringFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zStringFieldValue,
});
const zStringFieldInputTemplate = zFieldInputTemplateBase
  .extend({
    type: zStringFieldType,
    originalType: zFieldType.optional(),
    default: zStringFieldValue,
    maxLength: z.number().int().gte(0).optional(),
    minLength: z.number().int().gte(0).optional(),
  })
  .refine(
    (val) => {
      if (val.maxLength !== undefined && val.minLength !== undefined) {
        return val.maxLength >= val.minLength;
      }
      return true;
    },
    { message: 'maxLength must be greater than or equal to minLength' }
  );
const zStringFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zStringFieldType,
});
export type StringFieldValue = z.infer<typeof zStringFieldValue>;
export type StringFieldInputInstance = z.infer<typeof zStringFieldInputInstance>;
export type StringFieldInputTemplate = z.infer<typeof zStringFieldInputTemplate>;
export const isStringFieldInputInstance = buildInstanceTypeGuard(zStringFieldInputInstance);
export const isStringFieldInputTemplate = buildTemplateTypeGuard<StringFieldInputTemplate>('StringField', 'SINGLE');
// #endregion

// #region StringField Collection
export const zStringFieldCollectionValue = z.array(zStringFieldValue).optional();
const zStringFieldCollectionInputInstance = zFieldInputInstanceBase.extend({
  value: zStringFieldCollectionValue,
});
const zStringFieldCollectionInputTemplate = zFieldInputTemplateBase
  .extend({
    type: zStringCollectionFieldType,
    originalType: zFieldType.optional(),
    default: zStringFieldCollectionValue,
    maxLength: z.number().int().gte(0).optional(),
    minLength: z.number().int().gte(0).optional(),
    maxItems: z.number().int().gte(0).optional(),
    minItems: z.number().int().gte(0).optional(),
  })
  .refine(
    (val) => {
      if (val.maxLength !== undefined && val.minLength !== undefined) {
        return val.maxLength >= val.minLength;
      }
      return true;
    },
    { message: 'maxLength must be greater than or equal to minLength' }
  )
  .refine(
    (val) => {
      if (val.maxItems !== undefined && val.minItems !== undefined) {
        return val.maxItems >= val.minItems;
      }
      return true;
    },
    { message: 'maxItems must be greater than or equal to minItems' }
  );

const zStringFieldCollectionOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zStringCollectionFieldType,
});
export type StringFieldCollectionValue = z.infer<typeof zStringFieldCollectionValue>;
export type StringFieldCollectionInputInstance = z.infer<typeof zStringFieldCollectionInputInstance>;
export type StringFieldCollectionInputTemplate = z.infer<typeof zStringFieldCollectionInputTemplate>;
export const isStringFieldCollectionInputInstance = buildInstanceTypeGuard(zStringFieldCollectionInputInstance);
export const isStringFieldCollectionInputTemplate = buildTemplateTypeGuard<StringFieldCollectionInputTemplate>(
  'StringField',
  'COLLECTION'
);
// #endregion

// #region BooleanField
export const zBooleanFieldValue = z.boolean();
const zBooleanFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zBooleanFieldValue,
});
const zBooleanFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zBooleanFieldType,
  originalType: zFieldType.optional(),
  default: zBooleanFieldValue,
});
const zBooleanFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zBooleanFieldType,
});
export type BooleanFieldValue = z.infer<typeof zBooleanFieldValue>;
export type BooleanFieldInputInstance = z.infer<typeof zBooleanFieldInputInstance>;
export type BooleanFieldInputTemplate = z.infer<typeof zBooleanFieldInputTemplate>;
export const isBooleanFieldInputInstance = buildInstanceTypeGuard(zBooleanFieldInputInstance);
export const isBooleanFieldInputTemplate = buildTemplateTypeGuard<BooleanFieldInputTemplate>('BooleanField');
// #endregion

// #region EnumField
export const zEnumFieldValue = z.string();
const zEnumFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zEnumFieldValue,
});
const zEnumFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zEnumFieldType,
  originalType: zFieldType.optional(),
  default: zEnumFieldValue,
  options: z.array(z.string()),
  labels: z.record(z.string()).optional(),
});
const zEnumFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zEnumFieldType,
});
export type EnumFieldValue = z.infer<typeof zEnumFieldValue>;
export type EnumFieldInputInstance = z.infer<typeof zEnumFieldInputInstance>;
export type EnumFieldInputTemplate = z.infer<typeof zEnumFieldInputTemplate>;
export const isEnumFieldInputInstance = buildInstanceTypeGuard(zEnumFieldInputInstance);
export const isEnumFieldInputTemplate = buildTemplateTypeGuard<EnumFieldInputTemplate>('EnumField');
// #endregion

// #region ImageField
export const zImageFieldValue = zImageField.optional();
const zImageFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zImageFieldValue,
});
const zImageFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zImageFieldType,
  originalType: zFieldType.optional(),
  default: zImageFieldValue,
});
const zImageFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zImageFieldType,
});
export type ImageFieldValue = z.infer<typeof zImageFieldValue>;
export type ImageFieldInputInstance = z.infer<typeof zImageFieldInputInstance>;
export type ImageFieldInputTemplate = z.infer<typeof zImageFieldInputTemplate>;
export const isImageFieldInputInstance = buildInstanceTypeGuard(zImageFieldInputInstance);
export const isImageFieldInputTemplate = buildTemplateTypeGuard<ImageFieldInputTemplate>('ImageField', 'SINGLE');
// #endregion

// #region ImageField Collection
export const zImageFieldCollectionValue = z.array(zImageField).optional();
const zImageFieldCollectionInputInstance = zFieldInputInstanceBase.extend({
  value: zImageFieldCollectionValue,
});
const zImageFieldCollectionInputTemplate = zFieldInputTemplateBase
  .extend({
    type: zImageCollectionFieldType,
    originalType: zFieldType.optional(),
    default: zImageFieldCollectionValue,
    maxItems: z.number().int().gte(0).optional(),
    minItems: z.number().int().gte(0).optional(),
  })
  .refine(
    (val) => {
      if (val.maxItems !== undefined && val.minItems !== undefined) {
        return val.maxItems >= val.minItems;
      }
      return true;
    },
    { message: 'maxItems must be greater than or equal to minItems' }
  );

const zImageFieldCollectionOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zImageCollectionFieldType,
});
export type ImageFieldCollectionValue = z.infer<typeof zImageFieldCollectionValue>;
export type ImageFieldCollectionInputInstance = z.infer<typeof zImageFieldCollectionInputInstance>;
export type ImageFieldCollectionInputTemplate = z.infer<typeof zImageFieldCollectionInputTemplate>;
export const isImageFieldCollectionInputInstance = buildInstanceTypeGuard(zImageFieldCollectionInputInstance);
export const isImageFieldCollectionInputTemplate = buildTemplateTypeGuard<ImageFieldCollectionInputTemplate>(
  'ImageField',
  'COLLECTION'
);
// #endregion

// #region BoardField
export const zBoardFieldValue = zBoardField.optional();
const zBoardFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zBoardFieldValue,
});
const zBoardFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zBoardFieldType,
  originalType: zFieldType.optional(),
  default: zBoardFieldValue,
});
const zBoardFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zBoardFieldType,
});
export type BoardFieldValue = z.infer<typeof zBoardFieldValue>;
export type BoardFieldInputInstance = z.infer<typeof zBoardFieldInputInstance>;
export type BoardFieldInputTemplate = z.infer<typeof zBoardFieldInputTemplate>;
export const isBoardFieldInputInstance = buildInstanceTypeGuard(zBoardFieldInputInstance);
export const isBoardFieldInputTemplate = buildTemplateTypeGuard<BoardFieldInputTemplate>('BoardField');
// #endregion

// #region ColorField
export const zColorFieldValue = zColorField.optional();
const zColorFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zColorFieldValue,
});
const zColorFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zColorFieldType,
  originalType: zFieldType.optional(),
  default: zColorFieldValue,
});
const zColorFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zColorFieldType,
});
export type ColorFieldValue = z.infer<typeof zColorFieldValue>;
export type ColorFieldInputInstance = z.infer<typeof zColorFieldInputInstance>;
export type ColorFieldInputTemplate = z.infer<typeof zColorFieldInputTemplate>;
export const isColorFieldInputInstance = buildInstanceTypeGuard(zColorFieldInputInstance);
export const isColorFieldInputTemplate = buildTemplateTypeGuard<ColorFieldInputTemplate>('ColorField');
// #endregion

// #region MainModelField
export const zMainModelFieldValue = zModelIdentifierField.optional();
const zMainModelFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zMainModelFieldValue,
});
const zMainModelFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zMainModelFieldType,
  originalType: zFieldType.optional(),
  default: zMainModelFieldValue,
});
const zMainModelFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zMainModelFieldType,
});
export type MainModelFieldValue = z.infer<typeof zMainModelFieldValue>;
export type MainModelFieldInputInstance = z.infer<typeof zMainModelFieldInputInstance>;
export type MainModelFieldInputTemplate = z.infer<typeof zMainModelFieldInputTemplate>;
export const isMainModelFieldInputInstance = buildInstanceTypeGuard(zMainModelFieldInputInstance);
export const isMainModelFieldInputTemplate = buildTemplateTypeGuard<MainModelFieldInputTemplate>('MainModelField');
// #endregion

// #region ModelIdentifierField
export const zModelIdentifierFieldValue = zModelIdentifierField.optional();
const zModelIdentifierFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zModelIdentifierFieldValue,
});
const zModelIdentifierFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zModelIdentifierFieldType,
  originalType: zFieldType.optional(),
  default: zModelIdentifierFieldValue,
});
const zModelIdentifierFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zModelIdentifierFieldType,
});
export type ModelIdentifierFieldValue = z.infer<typeof zModelIdentifierFieldValue>;
export type ModelIdentifierFieldInputInstance = z.infer<typeof zModelIdentifierFieldInputInstance>;
export type ModelIdentifierFieldInputTemplate = z.infer<typeof zModelIdentifierFieldInputTemplate>;
export const isModelIdentifierFieldInputInstance = buildInstanceTypeGuard(zModelIdentifierFieldInputInstance);
export const isModelIdentifierFieldInputTemplate =
  buildTemplateTypeGuard<ModelIdentifierFieldInputTemplate>('ModelIdentifierField');
// #endregion

// #region SDXLMainModelField
const zSDXLMainModelFieldValue = zMainModelFieldValue; // TODO: Narrow to SDXL models only.
const zSDXLMainModelFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zSDXLMainModelFieldValue,
});
const zSDXLMainModelFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zSDXLMainModelFieldType,
  originalType: zFieldType.optional(),
  default: zSDXLMainModelFieldValue,
});
const zSDXLMainModelFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zSDXLMainModelFieldType,
});
export type SDXLMainModelFieldInputInstance = z.infer<typeof zSDXLMainModelFieldInputInstance>;
export type SDXLMainModelFieldInputTemplate = z.infer<typeof zSDXLMainModelFieldInputTemplate>;
export const isSDXLMainModelFieldInputInstance = buildInstanceTypeGuard(zSDXLMainModelFieldInputInstance);
export const isSDXLMainModelFieldInputTemplate =
  buildTemplateTypeGuard<SDXLMainModelFieldInputTemplate>('SDXLMainModelField');
// #endregion

// #region SD3MainModelField
const zSD3MainModelFieldValue = zMainModelFieldValue; // TODO: Narrow to SDXL models only.
const zSD3MainModelFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zSD3MainModelFieldValue,
});
const zSD3MainModelFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zSD3MainModelFieldType,
  originalType: zFieldType.optional(),
  default: zSD3MainModelFieldValue,
});
const zSD3MainModelFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zSD3MainModelFieldType,
});
export type SD3MainModelFieldInputInstance = z.infer<typeof zSD3MainModelFieldInputInstance>;
export type SD3MainModelFieldInputTemplate = z.infer<typeof zSD3MainModelFieldInputTemplate>;
export const isSD3MainModelFieldInputInstance = buildInstanceTypeGuard(zSD3MainModelFieldInputInstance);
export const isSD3MainModelFieldInputTemplate =
  buildTemplateTypeGuard<SD3MainModelFieldInputTemplate>('SD3MainModelField');
// #endregion

// #region FluxMainModelField
const zFluxMainModelFieldValue = zMainModelFieldValue; // TODO: Narrow to SDXL models only.
const zFluxMainModelFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zFluxMainModelFieldValue,
});
const zFluxMainModelFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zFluxMainModelFieldType,
  originalType: zFieldType.optional(),
  default: zFluxMainModelFieldValue,
});
const zFluxMainModelFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zFluxMainModelFieldType,
});
export type FluxMainModelFieldInputInstance = z.infer<typeof zFluxMainModelFieldInputInstance>;
export type FluxMainModelFieldInputTemplate = z.infer<typeof zFluxMainModelFieldInputTemplate>;
export const isFluxMainModelFieldInputInstance = buildInstanceTypeGuard(zFluxMainModelFieldInputInstance);
export const isFluxMainModelFieldInputTemplate =
  buildTemplateTypeGuard<FluxMainModelFieldInputTemplate>('FluxMainModelField');
// #endregion

// #region SDXLRefinerModelField
/** @alias */ // tells knip to ignore this duplicate export
export const zSDXLRefinerModelFieldValue = zMainModelFieldValue; // TODO: Narrow to SDXL Refiner models only.
const zSDXLRefinerModelFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zSDXLRefinerModelFieldValue,
});
const zSDXLRefinerModelFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zSDXLRefinerModelFieldType,
  originalType: zFieldType.optional(),
  default: zSDXLRefinerModelFieldValue,
});
const zSDXLRefinerModelFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zSDXLRefinerModelFieldType,
});
export type SDXLRefinerModelFieldValue = z.infer<typeof zSDXLRefinerModelFieldValue>;
export type SDXLRefinerModelFieldInputInstance = z.infer<typeof zSDXLRefinerModelFieldInputInstance>;
export type SDXLRefinerModelFieldInputTemplate = z.infer<typeof zSDXLRefinerModelFieldInputTemplate>;
export const isSDXLRefinerModelFieldInputInstance = buildInstanceTypeGuard(zSDXLRefinerModelFieldInputInstance);
export const isSDXLRefinerModelFieldInputTemplate =
  buildTemplateTypeGuard<SDXLRefinerModelFieldInputTemplate>('SDXLRefinerModelField');
// #endregion

// #region VAEModelField

export const zVAEModelFieldValue = zModelIdentifierField.optional();
const zVAEModelFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zVAEModelFieldValue,
});
const zVAEModelFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zVAEModelFieldType,
  originalType: zFieldType.optional(),
  default: zVAEModelFieldValue,
});
const zVAEModelFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zVAEModelFieldType,
});
export type VAEModelFieldValue = z.infer<typeof zVAEModelFieldValue>;
export type VAEModelFieldInputInstance = z.infer<typeof zVAEModelFieldInputInstance>;
export type VAEModelFieldInputTemplate = z.infer<typeof zVAEModelFieldInputTemplate>;
export const isVAEModelFieldInputInstance = buildInstanceTypeGuard(zVAEModelFieldInputInstance);
export const isVAEModelFieldInputTemplate = buildTemplateTypeGuard<VAEModelFieldInputTemplate>('VAEModelField');
// #endregion

// #region LoRAModelField
export const zLoRAModelFieldValue = zModelIdentifierField.optional();
const zLoRAModelFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zLoRAModelFieldValue,
});
const zLoRAModelFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zLoRAModelFieldType,
  originalType: zFieldType.optional(),
  default: zLoRAModelFieldValue,
});
const zLoRAModelFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zLoRAModelFieldType,
});
export type LoRAModelFieldValue = z.infer<typeof zLoRAModelFieldValue>;
export type LoRAModelFieldInputInstance = z.infer<typeof zLoRAModelFieldInputInstance>;
export type LoRAModelFieldInputTemplate = z.infer<typeof zLoRAModelFieldInputTemplate>;
export const isLoRAModelFieldInputInstance = buildInstanceTypeGuard(zLoRAModelFieldInputInstance);
export const isLoRAModelFieldInputTemplate = buildTemplateTypeGuard<LoRAModelFieldInputTemplate>('LoRAModelField');
// #endregion

// #region ControlNetModelField
export const zControlNetModelFieldValue = zModelIdentifierField.optional();
const zControlNetModelFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zControlNetModelFieldValue,
});
const zControlNetModelFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zControlNetModelFieldType,
  originalType: zFieldType.optional(),
  default: zControlNetModelFieldValue,
});
const zControlNetModelFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zControlNetModelFieldType,
});
export type ControlNetModelFieldValue = z.infer<typeof zControlNetModelFieldValue>;
export type ControlNetModelFieldInputInstance = z.infer<typeof zControlNetModelFieldInputInstance>;
export type ControlNetModelFieldInputTemplate = z.infer<typeof zControlNetModelFieldInputTemplate>;
export const isControlNetModelFieldInputInstance = buildInstanceTypeGuard(zControlNetModelFieldInputInstance);
export const isControlNetModelFieldInputTemplate =
  buildTemplateTypeGuard<ControlNetModelFieldInputTemplate>('ControlNetModelField');
// #endregion

// #region IPAdapterModelField
export const zIPAdapterModelFieldValue = zModelIdentifierField.optional();
const zIPAdapterModelFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zIPAdapterModelFieldValue,
});
const zIPAdapterModelFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zIPAdapterModelFieldType,
  originalType: zFieldType.optional(),
  default: zIPAdapterModelFieldValue,
});
const zIPAdapterModelFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zIPAdapterModelFieldType,
});
export type IPAdapterModelFieldValue = z.infer<typeof zIPAdapterModelFieldValue>;
export type IPAdapterModelFieldInputInstance = z.infer<typeof zIPAdapterModelFieldInputInstance>;
export type IPAdapterModelFieldInputTemplate = z.infer<typeof zIPAdapterModelFieldInputTemplate>;
export const isIPAdapterModelFieldInputInstance = buildInstanceTypeGuard(zIPAdapterModelFieldInputInstance);
export const isIPAdapterModelFieldInputTemplate =
  buildTemplateTypeGuard<IPAdapterModelFieldInputTemplate>('IPAdapterModelField');
// #endregion

// #region T2IAdapterField
export const zT2IAdapterModelFieldValue = zModelIdentifierField.optional();
const zT2IAdapterModelFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zT2IAdapterModelFieldValue,
});
const zT2IAdapterModelFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zT2IAdapterModelFieldType,
  originalType: zFieldType.optional(),
  default: zT2IAdapterModelFieldValue,
});
const zT2IAdapterModelFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zT2IAdapterModelFieldType,
});
export type T2IAdapterModelFieldValue = z.infer<typeof zT2IAdapterModelFieldValue>;
export type T2IAdapterModelFieldInputInstance = z.infer<typeof zT2IAdapterModelFieldInputInstance>;
export type T2IAdapterModelFieldInputTemplate = z.infer<typeof zT2IAdapterModelFieldInputTemplate>;
export const isT2IAdapterModelFieldInputInstance = buildInstanceTypeGuard(zT2IAdapterModelFieldInputInstance);
export const isT2IAdapterModelFieldInputTemplate =
  buildTemplateTypeGuard<T2IAdapterModelFieldInputTemplate>('T2IAdapterModelField');
// #endregion

// #region SpandrelModelToModelField
export const zSpandrelImageToImageModelFieldValue = zModelIdentifierField.optional();
const zSpandrelImageToImageModelFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zSpandrelImageToImageModelFieldValue,
});
const zSpandrelImageToImageModelFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zSpandrelImageToImageModelFieldType,
  originalType: zFieldType.optional(),
  default: zSpandrelImageToImageModelFieldValue,
});
const zSpandrelImageToImageModelFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zSpandrelImageToImageModelFieldType,
});
export type SpandrelImageToImageModelFieldValue = z.infer<typeof zSpandrelImageToImageModelFieldValue>;
export type SpandrelImageToImageModelFieldInputInstance = z.infer<typeof zSpandrelImageToImageModelFieldInputInstance>;
export type SpandrelImageToImageModelFieldInputTemplate = z.infer<typeof zSpandrelImageToImageModelFieldInputTemplate>;
export const isSpandrelImageToImageModelFieldInputInstance = buildInstanceTypeGuard(
  zSpandrelImageToImageModelFieldInputInstance
);
export const isSpandrelImageToImageModelFieldInputTemplate =
  buildTemplateTypeGuard<SpandrelImageToImageModelFieldInputTemplate>('SpandrelImageToImageModelField');
// #endregion

// #region T5EncoderModelField

export const zT5EncoderModelFieldValue = zModelIdentifierField.optional();
const zT5EncoderModelFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zT5EncoderModelFieldValue,
});
const zT5EncoderModelFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zT5EncoderModelFieldType,
  originalType: zFieldType.optional(),
  default: zT5EncoderModelFieldValue,
});
export type T5EncoderModelFieldValue = z.infer<typeof zT5EncoderModelFieldValue>;
export type T5EncoderModelFieldInputInstance = z.infer<typeof zT5EncoderModelFieldInputInstance>;
export type T5EncoderModelFieldInputTemplate = z.infer<typeof zT5EncoderModelFieldInputTemplate>;
export const isT5EncoderModelFieldInputInstance = buildInstanceTypeGuard(zT5EncoderModelFieldInputInstance);
export const isT5EncoderModelFieldInputTemplate =
  buildTemplateTypeGuard<T5EncoderModelFieldInputTemplate>('T5EncoderModelField');
// #endregion

// #region FluxVAEModelField
export const zFluxVAEModelFieldValue = zModelIdentifierField.optional();
const zFluxVAEModelFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zFluxVAEModelFieldValue,
});
const zFluxVAEModelFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zFluxVAEModelFieldType,
  originalType: zFieldType.optional(),
  default: zFluxVAEModelFieldValue,
});
export type FluxVAEModelFieldValue = z.infer<typeof zFluxVAEModelFieldValue>;
export type FluxVAEModelFieldInputInstance = z.infer<typeof zFluxVAEModelFieldInputInstance>;
export type FluxVAEModelFieldInputTemplate = z.infer<typeof zFluxVAEModelFieldInputTemplate>;
export const isFluxVAEModelFieldInputInstance = buildInstanceTypeGuard(zFluxVAEModelFieldInputInstance);
export const isFluxVAEModelFieldInputTemplate =
  buildTemplateTypeGuard<FluxVAEModelFieldInputTemplate>('FluxVAEModelField');
// #endregion

// #region CLIPEmbedModelField
export const zCLIPEmbedModelFieldValue = zModelIdentifierField.optional();
const zCLIPEmbedModelFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zCLIPEmbedModelFieldValue,
});
const zCLIPEmbedModelFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zCLIPEmbedModelFieldType,
  originalType: zFieldType.optional(),
  default: zCLIPEmbedModelFieldValue,
});
export type CLIPEmbedModelFieldValue = z.infer<typeof zCLIPEmbedModelFieldValue>;
export type CLIPEmbedModelFieldInputInstance = z.infer<typeof zCLIPEmbedModelFieldInputInstance>;
export type CLIPEmbedModelFieldInputTemplate = z.infer<typeof zCLIPEmbedModelFieldInputTemplate>;
export const isCLIPEmbedModelFieldInputInstance = buildInstanceTypeGuard(zCLIPEmbedModelFieldInputInstance);
export const isCLIPEmbedModelFieldInputTemplate =
  buildTemplateTypeGuard<CLIPEmbedModelFieldInputTemplate>('CLIPEmbedModelField');
// #endregion

// #region CLIPLEmbedModelField
export const zCLIPLEmbedModelFieldValue = zModelIdentifierField.optional();
const zCLIPLEmbedModelFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zCLIPLEmbedModelFieldValue,
});
const zCLIPLEmbedModelFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zCLIPLEmbedModelFieldType,
  originalType: zFieldType.optional(),
  default: zCLIPLEmbedModelFieldValue,
});
export type CLIPLEmbedModelFieldValue = z.infer<typeof zCLIPLEmbedModelFieldValue>;
export type CLIPLEmbedModelFieldInputInstance = z.infer<typeof zCLIPLEmbedModelFieldInputInstance>;
export type CLIPLEmbedModelFieldInputTemplate = z.infer<typeof zCLIPLEmbedModelFieldInputTemplate>;
export const isCLIPLEmbedModelFieldInputInstance = buildInstanceTypeGuard(zCLIPLEmbedModelFieldInputInstance);
export const isCLIPLEmbedModelFieldInputTemplate =
  buildTemplateTypeGuard<CLIPLEmbedModelFieldInputTemplate>('CLIPLEmbedModelField');
// #endregion

// #region CLIPGEmbedModelField
export const zCLIPGEmbedModelFieldValue = zModelIdentifierField.optional();
const zCLIPGEmbedModelFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zCLIPGEmbedModelFieldValue,
});
const zCLIPGEmbedModelFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zCLIPGEmbedModelFieldType,
  originalType: zFieldType.optional(),
  default: zCLIPGEmbedModelFieldValue,
});
export type CLIPGEmbedModelFieldValue = z.infer<typeof zCLIPLEmbedModelFieldValue>;
export type CLIPGEmbedModelFieldInputInstance = z.infer<typeof zCLIPGEmbedModelFieldInputInstance>;
export type CLIPGEmbedModelFieldInputTemplate = z.infer<typeof zCLIPGEmbedModelFieldInputTemplate>;
export const isCLIPGEmbedModelFieldInputInstance = buildInstanceTypeGuard(zCLIPGEmbedModelFieldInputInstance);
export const isCLIPGEmbedModelFieldInputTemplate =
  buildTemplateTypeGuard<CLIPGEmbedModelFieldInputTemplate>('CLIPGEmbedModelField');
// #endregion

// #region ControlLoRAModelField
export const zControlLoRAModelFieldValue = zModelIdentifierField.optional();
const zControlLoRAModelFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zControlLoRAModelFieldValue,
});
const zControlLoRAModelFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zControlLoRAModelFieldType,
  originalType: zFieldType.optional(),
  default: zControlLoRAModelFieldValue,
});
export type ControlLoRAModelFieldValue = z.infer<typeof zCLIPLEmbedModelFieldValue>;
export type ControlLoRAModelFieldInputInstance = z.infer<typeof zControlLoRAModelFieldInputInstance>;
export type ControlLoRAModelFieldInputTemplate = z.infer<typeof zControlLoRAModelFieldInputTemplate>;
export const isControlLoRAModelFieldInputInstance = buildInstanceTypeGuard(zControlLoRAModelFieldInputInstance);
export const isControlLoRAModelFieldInputTemplate =
  buildTemplateTypeGuard<ControlLoRAModelFieldInputTemplate>('ControlLoRAModelField');
// #endregion

// #region SchedulerField
export const zSchedulerFieldValue = zSchedulerField.optional();
const zSchedulerFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zSchedulerFieldValue,
});
const zSchedulerFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zSchedulerFieldType,
  originalType: zFieldType.optional(),
  default: zSchedulerFieldValue,
});
const zSchedulerFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zSchedulerFieldType,
});
export type SchedulerFieldValue = z.infer<typeof zSchedulerFieldValue>;
export type SchedulerFieldInputInstance = z.infer<typeof zSchedulerFieldInputInstance>;
export type SchedulerFieldInputTemplate = z.infer<typeof zSchedulerFieldInputTemplate>;
export const isSchedulerFieldInputInstance = buildInstanceTypeGuard(zSchedulerFieldInputInstance);
export const isSchedulerFieldInputTemplate = buildTemplateTypeGuard<SchedulerFieldInputTemplate>('SchedulerField');
// #endregion

// #region FloatGeneratorField
export const FloatGeneratorArithmeticSequenceType = 'float_generator_arithmetic_sequence';
const zFloatGeneratorArithmeticSequence = z.object({
  type: z.literal(FloatGeneratorArithmeticSequenceType).default(FloatGeneratorArithmeticSequenceType),
  start: z.number().default(0),
  step: z.number().default(0.1),
  count: z.number().int().default(10),
  values: z.array(z.number()).nullish(),
});
export type FloatGeneratorArithmeticSequence = z.infer<typeof zFloatGeneratorArithmeticSequence>;
export const getFloatGeneratorArithmeticSequenceDefaults = () => zFloatGeneratorArithmeticSequence.parse({});
const getFloatGeneratorArithmeticSequenceValues = (generator: FloatGeneratorArithmeticSequence) => {
  const { start, step, count } = generator;
  if (step === 0) {
    return [start];
  }
  const values = Array.from({ length: count }, (_, i) => start + i * step);
  return values;
};

export const FloatGeneratorLinearDistributionType = 'float_generator_linear_distribution';
const zFloatGeneratorLinearDistribution = z.object({
  type: z.literal(FloatGeneratorLinearDistributionType).default(FloatGeneratorLinearDistributionType),
  start: z.number().default(0),
  end: z.number().default(1),
  count: z.number().int().default(10),
  values: z.array(z.number()).nullish(),
});
export type FloatGeneratorLinearDistribution = z.infer<typeof zFloatGeneratorLinearDistribution>;
const getFloatGeneratorLinearDistributionDefaults = () => zFloatGeneratorLinearDistribution.parse({});
const getFloatGeneratorLinearDistributionValues = (generator: FloatGeneratorLinearDistribution) => {
  const { start, end, count } = generator;
  if (count === 1) {
    return [start];
  }
  const values = Array.from({ length: count }, (_, i) => start + (end - start) * (i / (count - 1)));
  return values;
};

export const FloatGeneratorUniformRandomDistributionType = 'float_generator_random_distribution_uniform';
const zFloatGeneratorUniformRandomDistribution = z.object({
  type: z.literal(FloatGeneratorUniformRandomDistributionType).default(FloatGeneratorUniformRandomDistributionType),
  min: z.number().default(0),
  max: z.number().default(1),
  count: z.number().int().default(10),
  seed: z.number().int().nullish(),
  values: z.array(z.number()).nullish(),
});
export type FloatGeneratorUniformRandomDistribution = z.infer<typeof zFloatGeneratorUniformRandomDistribution>;
const getFloatGeneratorUniformRandomDistributionDefaults = () => zFloatGeneratorUniformRandomDistribution.parse({});
const getRng = (seed?: number | null) => {
  if (isNil(seed)) {
    return () => Math.random();
  }
  const m = new MersenneTwister(seed);
  return () => m.random();
};
const getFloatGeneratorUniformRandomDistributionValues = (generator: FloatGeneratorUniformRandomDistribution) => {
  const { min, max, count, seed } = generator;
  const rng = getRng(seed);
  const values = Array.from({ length: count }, (_) => rng() * (max - min) + min);
  return values;
};

export const FloatGeneratorParseStringType = 'float_generator_parse_string';
const zFloatGeneratorParseString = z.object({
  type: z.literal(FloatGeneratorParseStringType).default(FloatGeneratorParseStringType),
  input: z.string().default('0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1'),
  splitOn: z.string().default(','),
  values: z.array(z.number()).nullish(),
});
export type FloatGeneratorParseString = z.infer<typeof zFloatGeneratorParseString>;
const getFloatGeneratorParseStringDefaults = () => zFloatGeneratorParseString.parse({});
const getFloatGeneratorParseStringValues = (generator: FloatGeneratorParseString) => {
  const { input, splitOn } = generator;

  let splitValues: string[] = [];
  if (splitOn === '') {
    // special case for empty splitOn
    splitValues = [input];
  } else {
    // try to parse splitOn as a JSON string, this allows for special characters like \n
    try {
      splitValues = input.split(JSON.parse(`"${splitOn}"`));
    } catch {
      // if JSON parsing fails, just split on the string
      splitValues = input.split(splitOn);
    }
  }
  const values = splitValues
    .map(trim)
    .filter((s) => s.length > 0)
    .map((s) => parseFloat(s))
    .filter((n) => !isNaN(n));

  return values;
};

export const zFloatGeneratorFieldValue = z.union([
  zFloatGeneratorArithmeticSequence,
  zFloatGeneratorLinearDistribution,
  zFloatGeneratorUniformRandomDistribution,
  zFloatGeneratorParseString,
]);
const zFloatGeneratorFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zFloatGeneratorFieldValue,
});
const zFloatGeneratorFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zFloatGeneratorFieldType,
  originalType: zFieldType.optional(),
  default: zFloatGeneratorFieldValue,
});
const zFloatGeneratorFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zFloatGeneratorFieldType,
});
export type FloatGeneratorFieldValue = z.infer<typeof zFloatGeneratorFieldValue>;
export type FloatGeneratorFieldInputInstance = z.infer<typeof zFloatGeneratorFieldInputInstance>;
export type FloatGeneratorFieldInputTemplate = z.infer<typeof zFloatGeneratorFieldInputTemplate>;
export const isFloatGeneratorFieldInputInstance = buildInstanceTypeGuard(zFloatGeneratorFieldInputInstance);
export const isFloatGeneratorFieldInputTemplate =
  buildTemplateTypeGuard<FloatGeneratorFieldInputTemplate>('FloatGeneratorField');
export const resolveFloatGeneratorField = ({ value }: FloatGeneratorFieldInputInstance) => {
  if (value.values) {
    return value.values;
  }
  if (value.type === FloatGeneratorArithmeticSequenceType) {
    return getFloatGeneratorArithmeticSequenceValues(value);
  }
  if (value.type === FloatGeneratorLinearDistributionType) {
    return getFloatGeneratorLinearDistributionValues(value);
  }
  if (value.type === FloatGeneratorUniformRandomDistributionType) {
    return getFloatGeneratorUniformRandomDistributionValues(value);
  }
  if (value.type === FloatGeneratorParseStringType) {
    return getFloatGeneratorParseStringValues(value);
  }
  assert(false, 'Invalid float generator type');
};

export const getFloatGeneratorDefaults = (type: FloatGeneratorFieldValue['type']) => {
  if (type === FloatGeneratorArithmeticSequenceType) {
    return getFloatGeneratorArithmeticSequenceDefaults();
  }
  if (type === FloatGeneratorLinearDistributionType) {
    return getFloatGeneratorLinearDistributionDefaults();
  }
  if (type === FloatGeneratorUniformRandomDistributionType) {
    return getFloatGeneratorUniformRandomDistributionDefaults();
  }
  if (type === FloatGeneratorParseStringType) {
    return getFloatGeneratorParseStringDefaults();
  }
  assert(false, 'Invalid float generator type');
};

// #endregion

// #region IntegerGeneratorField
export const IntegerGeneratorArithmeticSequenceType = 'integer_generator_arithmetic_sequence';
const zIntegerGeneratorArithmeticSequence = z.object({
  type: z.literal(IntegerGeneratorArithmeticSequenceType).default(IntegerGeneratorArithmeticSequenceType),
  start: z.number().int().default(0),
  step: z.number().int().default(1),
  count: z.number().int().default(10),
  values: z.array(z.number().int()).nullish(),
});
export type IntegerGeneratorArithmeticSequence = z.infer<typeof zIntegerGeneratorArithmeticSequence>;
export const getIntegerGeneratorArithmeticSequenceDefaults = () => zIntegerGeneratorArithmeticSequence.parse({});
const getIntegerGeneratorArithmeticSequenceValues = (generator: IntegerGeneratorArithmeticSequence) => {
  const { start, step, count } = generator;
  if (step === 0) {
    return [start];
  }
  const values = Array.from({ length: count }, (_, i) => start + i * step);
  return values;
};

export const IntegerGeneratorLinearDistributionType = 'integer_generator_linear_distribution';
const zIntegerGeneratorLinearDistribution = z.object({
  type: z.literal(IntegerGeneratorLinearDistributionType).default(IntegerGeneratorLinearDistributionType),
  start: z.number().int().default(0),
  end: z.number().int().default(10),
  count: z.number().int().default(10),
  values: z.array(z.number().int()).nullish(),
});
export type IntegerGeneratorLinearDistribution = z.infer<typeof zIntegerGeneratorLinearDistribution>;
const getIntegerGeneratorLinearDistributionDefaults = () => zIntegerGeneratorLinearDistribution.parse({});
const getIntegerGeneratorLinearDistributionValues = (generator: IntegerGeneratorLinearDistribution) => {
  const { start, end, count } = generator;
  if (count === 1) {
    return [start];
  }
  const values = Array.from({ length: count }, (_, i) => start + Math.round((end - start) * (i / (count - 1))));
  return values;
};

export const IntegerGeneratorUniformRandomDistributionType = 'integer_generator_random_distribution_uniform';
const zIntegerGeneratorUniformRandomDistribution = z.object({
  type: z.literal(IntegerGeneratorUniformRandomDistributionType).default(IntegerGeneratorUniformRandomDistributionType),
  min: z.number().int().default(0),
  max: z.number().int().default(10),
  count: z.number().int().default(10),
  seed: z.number().int().nullish(),
  values: z.array(z.number().int()).nullish(),
});
export type IntegerGeneratorUniformRandomDistribution = z.infer<typeof zIntegerGeneratorUniformRandomDistribution>;
const getIntegerGeneratorUniformRandomDistributionDefaults = () => zIntegerGeneratorUniformRandomDistribution.parse({});
const getIntegerGeneratorUniformRandomDistributionValues = (generator: IntegerGeneratorUniformRandomDistribution) => {
  const { min, max, count, seed } = generator;
  const rng = getRng(seed);
  const values = Array.from({ length: count }, () => Math.floor(rng() * (max - min + 1)) + min);
  return values;
};

export const IntegerGeneratorParseStringType = 'integer_generator_parse_string';
const zIntegerGeneratorParseString = z.object({
  type: z.literal(IntegerGeneratorParseStringType).default(IntegerGeneratorParseStringType),
  input: z.string().default('1,2,3,4,5,6,7,8,9,10'),
  splitOn: z.string().default(','),
  values: z.array(z.number().int()).nullish(),
});
export type IntegerGeneratorParseString = z.infer<typeof zIntegerGeneratorParseString>;
const getIntegerGeneratorParseStringDefaults = () => zIntegerGeneratorParseString.parse({});
const getIntegerGeneratorParseStringValues = (generator: IntegerGeneratorParseString) => {
  const { input, splitOn } = generator;

  let splitValues: string[] = [];
  if (splitOn === '') {
    // special case for empty splitOn
    splitValues = [input];
  } else {
    // try to parse splitOn as a JSON string, this allows for special characters like \n
    try {
      splitValues = input.split(JSON.parse(`"${splitOn}"`));
    } catch {
      // if JSON parsing fails, just split on the string
      splitValues = input.split(splitOn);
    }
  }

  const values = splitValues
    .map(trim)
    .filter((s) => s.length > 0)
    .map((s) => parseInt(s, 10))
    .filter((n) => !isNaN(n));

  return values;
};

export const zIntegerGeneratorFieldValue = z.union([
  zIntegerGeneratorArithmeticSequence,
  zIntegerGeneratorLinearDistribution,
  zIntegerGeneratorUniformRandomDistribution,
  zIntegerGeneratorParseString,
]);
const zIntegerGeneratorFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zIntegerGeneratorFieldValue,
});
const zIntegerGeneratorFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zIntegerGeneratorFieldType,
  originalType: zFieldType.optional(),
  default: zIntegerGeneratorFieldValue,
});
const zIntegerGeneratorFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zIntegerGeneratorFieldType,
});
export type IntegerGeneratorFieldValue = z.infer<typeof zIntegerGeneratorFieldValue>;
export type IntegerGeneratorFieldInputInstance = z.infer<typeof zIntegerGeneratorFieldInputInstance>;
export type IntegerGeneratorFieldInputTemplate = z.infer<typeof zIntegerGeneratorFieldInputTemplate>;
export const isIntegerGeneratorFieldInputInstance = buildInstanceTypeGuard(zIntegerGeneratorFieldInputInstance);
export const isIntegerGeneratorFieldInputTemplate =
  buildTemplateTypeGuard<IntegerGeneratorFieldInputTemplate>('IntegerGeneratorField');
export const resolveIntegerGeneratorField = ({ value }: IntegerGeneratorFieldInputInstance) => {
  if (value.values) {
    return value.values;
  }
  if (value.type === IntegerGeneratorArithmeticSequenceType) {
    return getIntegerGeneratorArithmeticSequenceValues(value);
  }
  if (value.type === IntegerGeneratorLinearDistributionType) {
    return getIntegerGeneratorLinearDistributionValues(value);
  }
  if (value.type === IntegerGeneratorUniformRandomDistributionType) {
    return getIntegerGeneratorUniformRandomDistributionValues(value);
  }
  if (value.type === IntegerGeneratorParseStringType) {
    return getIntegerGeneratorParseStringValues(value);
  }
  assert(false, 'Invalid integer generator type');
};
export const getIntegerGeneratorDefaults = (type: IntegerGeneratorFieldValue['type']) => {
  if (type === IntegerGeneratorArithmeticSequenceType) {
    return getIntegerGeneratorArithmeticSequenceDefaults();
  }
  if (type === IntegerGeneratorLinearDistributionType) {
    return getIntegerGeneratorLinearDistributionDefaults();
  }
  if (type === IntegerGeneratorUniformRandomDistributionType) {
    return getIntegerGeneratorUniformRandomDistributionDefaults();
  }
  if (type === IntegerGeneratorParseStringType) {
    return getIntegerGeneratorParseStringDefaults();
  }
  assert(false, 'Invalid integer generator type');
};
// #endregion

// #region StringGeneratorField
export const StringGeneratorParseStringType = 'string_generator_parse_string';
const zStringGeneratorParseString = z.object({
  type: z.literal(StringGeneratorParseStringType).default(StringGeneratorParseStringType),
  input: z.string().default('foo,bar,baz,qux'),
  splitOn: z.string().default(','),
  values: z.array(z.string()).nullish(),
});
export type StringGeneratorParseString = z.infer<typeof zStringGeneratorParseString>;
export const getStringGeneratorParseStringDefaults = () => zStringGeneratorParseString.parse({});
const getStringGeneratorParseStringValues = (generator: StringGeneratorParseString) => {
  const { input, splitOn } = generator;
  let splitValues: string[] = [];
  if (splitOn === '') {
    // special case for empty splitOn
    splitValues = [input];
  } else {
    // try to parse splitOn as a JSON string, this allows for special characters like \n
    try {
      splitValues = input.split(JSON.parse(`"${splitOn}"`));
    } catch {
      // if JSON parsing fails, just split on the string
      splitValues = input.split(splitOn);
    }
  }
  const values = splitValues.filter((s) => s.length > 0);
  return values;
};

export const StringGeneratorDynamicPromptsCombinatorialType = 'string_generator_dynamic_prompts_combinatorial';
const zStringGeneratorDynamicPromptsCombinatorial = z.object({
  type: z
    .literal(StringGeneratorDynamicPromptsCombinatorialType)
    .default(StringGeneratorDynamicPromptsCombinatorialType),
  input: z.string().default('a super {cute|ferocious} {dog|cat}'),
  maxPrompts: z.number().int().gte(1).default(10),
  values: z.array(z.string()).nullish(),
});
export type StringGeneratorDynamicPromptsCombinatorial = z.infer<typeof zStringGeneratorDynamicPromptsCombinatorial>;
const getStringGeneratorDynamicPromptsCombinatorialDefaults = () =>
  zStringGeneratorDynamicPromptsCombinatorial.parse({});
const getStringGeneratorDynamicPromptsCombinatorialValues = (generator: StringGeneratorDynamicPromptsCombinatorial) => {
  const { values } = generator;
  return values ?? [];
};

export const StringGeneratorDynamicPromptsRandomType = 'string_generator_dynamic_prompts_random';
const zStringGeneratorDynamicPromptsRandom = z.object({
  type: z.literal(StringGeneratorDynamicPromptsRandomType).default(StringGeneratorDynamicPromptsRandomType),
  input: z.string().default('a super {cute|ferocious} {dog|cat}'),
  count: z.number().int().gte(1).default(10),
  seed: z.number().int().nullish(),
  values: z.array(z.string()).nullish(),
});
export type StringGeneratorDynamicPromptsRandom = z.infer<typeof zStringGeneratorDynamicPromptsRandom>;
const getStringGeneratorDynamicPromptsRandomDefaults = () => zStringGeneratorDynamicPromptsRandom.parse({});
const getStringGeneratorDynamicPromptsRandomValues = (generator: StringGeneratorDynamicPromptsRandom) => {
  const { values } = generator;
  return values ?? [];
};

export const zStringGeneratorFieldValue = z.union([
  zStringGeneratorParseString,
  zStringGeneratorDynamicPromptsCombinatorial,
  zStringGeneratorDynamicPromptsRandom,
]);
const zStringGeneratorFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zStringGeneratorFieldValue,
});
const zStringGeneratorFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zStringGeneratorFieldType,
  originalType: zFieldType.optional(),
  default: zStringGeneratorFieldValue,
});
const zStringGeneratorFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zStringGeneratorFieldType,
});
export type StringGeneratorFieldValue = z.infer<typeof zStringGeneratorFieldValue>;
export type StringGeneratorFieldInputInstance = z.infer<typeof zStringGeneratorFieldInputInstance>;
export type StringGeneratorFieldInputTemplate = z.infer<typeof zStringGeneratorFieldInputTemplate>;
export const isStringGeneratorFieldInputInstance = buildInstanceTypeGuard(zStringGeneratorFieldInputInstance);
export const isStringGeneratorFieldInputTemplate = buildTemplateTypeGuard<StringGeneratorFieldInputTemplate>(
  zStringGeneratorFieldType.shape.name.value
);

export const resolveStringGeneratorField = ({ value }: StringGeneratorFieldInputInstance) => {
  if (value.values) {
    return value.values;
  }
  if (value.type === StringGeneratorParseStringType) {
    return getStringGeneratorParseStringValues(value);
  }
  if (value.type === StringGeneratorDynamicPromptsRandomType) {
    return getStringGeneratorDynamicPromptsRandomValues(value);
  }
  if (value.type === StringGeneratorDynamicPromptsCombinatorialType) {
    return getStringGeneratorDynamicPromptsCombinatorialValues(value);
  }
  assert(false, 'Invalid string generator type');
};
export const getStringGeneratorDefaults = (type: StringGeneratorFieldValue['type']) => {
  if (type === StringGeneratorParseStringType) {
    return getStringGeneratorParseStringDefaults();
  }
  if (type === StringGeneratorDynamicPromptsRandomType) {
    return getStringGeneratorDynamicPromptsRandomDefaults();
  }
  if (type === StringGeneratorDynamicPromptsCombinatorialType) {
    return getStringGeneratorDynamicPromptsCombinatorialDefaults();
  }
  assert(false, 'Invalid string generator type');
};
// #endregion

// #region StatelessField
/**
 * StatelessField is a catchall for stateless fields with no UI input components. They do not
 * do not support "direct" input, instead only accepting connections from other fields.
 *
 * This field type serves as a "generic" field type.
 *
 * Examples include:
 * - Fields like UNetField or LatentsField where we do not allow direct UI input
 * - Reserved fields like IsIntermediate
 * - Any other field we don't have full-on schemas for
 */

const zStatelessFieldValue = z.undefined().catch(undefined); // stateless --> no value, but making this z.never() introduces a lot of extra TS fanagling
const zStatelessFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zStatelessFieldValue,
});
const zStatelessFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zStatelessFieldType,
  originalType: zFieldType.optional(),
  default: zStatelessFieldValue,
  input: z.literal('connection'), // stateless --> only accepts connection inputs
});
const zStatelessFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zStatelessFieldType,
});

export type StatelessFieldInputTemplate = z.infer<typeof zStatelessFieldInputTemplate>;
// #endregion

/**
 * Here we define the main field unions:
 * - FieldType
 * - FieldValue
 * - FieldInputInstance
 * - FieldOutputInstance
 * - FieldInputTemplate
 * - FieldOutputTemplate
 *
 * All stateful fields are unioned together, and then that union is unioned with StatelessField.
 *
 * This allows us to interact with stateful fields without needing to worry about "generic" handling
 * for all other StatelessFields.
 */

// #region StatefulFieldValue & FieldValue
export const zStatefulFieldValue = z.union([
  zIntegerFieldValue,
  zIntegerFieldCollectionValue,
  zFloatFieldValue,
  zFloatFieldCollectionValue,
  zStringFieldValue,
  zStringFieldCollectionValue,
  zBooleanFieldValue,
  zEnumFieldValue,
  zImageFieldValue,
  zImageFieldCollectionValue,
  zBoardFieldValue,
  zModelIdentifierFieldValue,
  zMainModelFieldValue,
  zSDXLMainModelFieldValue,
  zFluxMainModelFieldValue,
  zSD3MainModelFieldValue,
  zSDXLRefinerModelFieldValue,
  zVAEModelFieldValue,
  zLoRAModelFieldValue,
  zControlNetModelFieldValue,
  zIPAdapterModelFieldValue,
  zT2IAdapterModelFieldValue,
  zSpandrelImageToImageModelFieldValue,
  zT5EncoderModelFieldValue,
  zFluxVAEModelFieldValue,
  zCLIPEmbedModelFieldValue,
  zCLIPLEmbedModelFieldValue,
  zCLIPGEmbedModelFieldValue,
  zControlLoRAModelFieldValue,
  zColorFieldValue,
  zSchedulerFieldValue,
  zFloatGeneratorFieldValue,
  zIntegerGeneratorFieldValue,
  zStringGeneratorFieldValue,
]);
export type StatefulFieldValue = z.infer<typeof zStatefulFieldValue>;

const zFieldValue = z.union([zStatefulFieldValue, zStatelessFieldValue]);
export type FieldValue = z.infer<typeof zFieldValue>;
// #endregion

// #region StatefulFieldInputInstance & FieldInputInstance
const zStatefulFieldInputInstance = z.union([
  zIntegerFieldInputInstance,
  zIntegerFieldCollectionInputInstance,
  zFloatFieldInputInstance,
  zFloatFieldCollectionInputInstance,
  zStringFieldInputInstance,
  zStringFieldCollectionInputInstance,
  zBooleanFieldInputInstance,
  zEnumFieldInputInstance,
  zImageFieldInputInstance,
  zImageFieldCollectionInputInstance,
  zBoardFieldInputInstance,
  zModelIdentifierFieldInputInstance,
  zMainModelFieldInputInstance,
  zFluxMainModelFieldInputInstance,
  zSD3MainModelFieldInputInstance,
  zSDXLMainModelFieldInputInstance,
  zSDXLRefinerModelFieldInputInstance,
  zVAEModelFieldInputInstance,
  zLoRAModelFieldInputInstance,
  zControlNetModelFieldInputInstance,
  zIPAdapterModelFieldInputInstance,
  zT2IAdapterModelFieldInputInstance,
  zSpandrelImageToImageModelFieldInputInstance,
  zT5EncoderModelFieldInputInstance,
  zFluxVAEModelFieldInputInstance,
  zCLIPEmbedModelFieldInputInstance,
  zColorFieldInputInstance,
  zSchedulerFieldInputInstance,
  zFloatGeneratorFieldInputInstance,
  zIntegerGeneratorFieldInputInstance,
  zStringGeneratorFieldInputInstance,
]);

export const zFieldInputInstance = z.union([zStatefulFieldInputInstance, zStatelessFieldInputInstance]);
export type FieldInputInstance = z.infer<typeof zFieldInputInstance>;
// #endregion

// #region StatefulFieldInputTemplate & FieldInputTemplate
const zStatefulFieldInputTemplate = z.union([
  zIntegerFieldInputTemplate,
  zIntegerFieldCollectionInputTemplate,
  zFloatFieldInputTemplate,
  zFloatFieldCollectionInputTemplate,
  zStringFieldInputTemplate,
  zStringFieldCollectionInputTemplate,
  zBooleanFieldInputTemplate,
  zEnumFieldInputTemplate,
  zImageFieldInputTemplate,
  zImageFieldCollectionInputTemplate,
  zBoardFieldInputTemplate,
  zModelIdentifierFieldInputTemplate,
  zMainModelFieldInputTemplate,
  zFluxMainModelFieldInputTemplate,
  zSD3MainModelFieldInputTemplate,
  zSDXLMainModelFieldInputTemplate,
  zSDXLRefinerModelFieldInputTemplate,
  zVAEModelFieldInputTemplate,
  zLoRAModelFieldInputTemplate,
  zControlNetModelFieldInputTemplate,
  zIPAdapterModelFieldInputTemplate,
  zT2IAdapterModelFieldInputTemplate,
  zSpandrelImageToImageModelFieldInputTemplate,
  zT5EncoderModelFieldInputTemplate,
  zFluxVAEModelFieldInputTemplate,
  zCLIPEmbedModelFieldInputTemplate,
  zCLIPLEmbedModelFieldInputTemplate,
  zCLIPGEmbedModelFieldInputTemplate,
  zControlLoRAModelFieldInputTemplate,
  zColorFieldInputTemplate,
  zSchedulerFieldInputTemplate,
  zStatelessFieldInputTemplate,
  zFloatGeneratorFieldInputTemplate,
  zIntegerGeneratorFieldInputTemplate,
  zStringGeneratorFieldInputTemplate,
]);

export const zFieldInputTemplate = z.union([zStatefulFieldInputTemplate, zStatelessFieldInputTemplate]);
export type FieldInputTemplate = z.infer<typeof zFieldInputTemplate>;
// #endregion

// #region StatefulFieldOutputTemplate & FieldOutputTemplate
const zStatefulFieldOutputTemplate = z.union([
  zIntegerFieldOutputTemplate,
  zIntegerFieldCollectionOutputTemplate,
  zFloatFieldOutputTemplate,
  zFloatFieldCollectionOutputTemplate,
  zStringFieldOutputTemplate,
  zStringFieldCollectionOutputTemplate,
  zBooleanFieldOutputTemplate,
  zEnumFieldOutputTemplate,
  zImageFieldOutputTemplate,
  zImageFieldCollectionOutputTemplate,
  zBoardFieldOutputTemplate,
  zModelIdentifierFieldOutputTemplate,
  zMainModelFieldOutputTemplate,
  zFluxMainModelFieldOutputTemplate,
  zSD3MainModelFieldOutputTemplate,
  zSDXLMainModelFieldOutputTemplate,
  zSDXLRefinerModelFieldOutputTemplate,
  zVAEModelFieldOutputTemplate,
  zLoRAModelFieldOutputTemplate,
  zControlNetModelFieldOutputTemplate,
  zIPAdapterModelFieldOutputTemplate,
  zT2IAdapterModelFieldOutputTemplate,
  zSpandrelImageToImageModelFieldOutputTemplate,
  zColorFieldOutputTemplate,
  zSchedulerFieldOutputTemplate,
  zFloatGeneratorFieldOutputTemplate,
  zIntegerGeneratorFieldOutputTemplate,
  zStringGeneratorFieldOutputTemplate,
]);

export const zFieldOutputTemplate = z.union([zStatefulFieldOutputTemplate, zStatelessFieldOutputTemplate]);
export type FieldOutputTemplate = z.infer<typeof zFieldOutputTemplate>;
// #endregion

// #region FieldInputTemplate Type Guards

// #endregion
