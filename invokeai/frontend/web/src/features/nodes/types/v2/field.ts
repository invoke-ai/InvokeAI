import { z } from 'zod';

import {
  zBoardField,
  zColorField,
  zControlNetModelField,
  zImageField,
  zIPAdapterModelField,
  zLoRAModelField,
  zMainModelField,
  zSchedulerField,
  zT2IAdapterModelField,
  zVAEModelField,
} from './common';

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
export const zFieldInput = z.enum(['connection', 'direct', 'any']);
export type FieldInput = z.infer<typeof zFieldInput>;

export const zFieldUIComponent = z.enum(['none', 'textarea', 'slider']);
export type FieldUIComponent = z.infer<typeof zFieldUIComponent>;

export const zFieldInstanceBase = z.object({
  id: z.string().trim().min(1),
  name: z.string().trim().min(1),
});
export const zFieldInputInstanceBase = zFieldInstanceBase.extend({
  fieldKind: z.literal('input'),
  label: z.string().nullish(),
});
export const zFieldOutputInstanceBase = zFieldInstanceBase.extend({
  fieldKind: z.literal('output'),
});
export type FieldInstanceBase = z.infer<typeof zFieldInstanceBase>;
export type FieldInputInstanceBase = z.infer<typeof zFieldInputInstanceBase>;
export type FieldOutputInstanceBase = z.infer<typeof zFieldOutputInstanceBase>;

export const zFieldTemplateBase = z.object({
  name: z.string().min(1),
  title: z.string().min(1),
  description: z.string().nullish(),
  ui_hidden: z.boolean(),
  ui_type: z.string().nullish(),
  ui_order: z.number().int().nullish(),
});
export const zFieldInputTemplateBase = zFieldTemplateBase.extend({
  fieldKind: z.literal('input'),
  input: zFieldInput,
  required: z.boolean(),
  ui_component: zFieldUIComponent.nullish(),
  ui_choice_labels: z.record(z.string()).nullish(),
});
export const zFieldOutputTemplateBase = zFieldTemplateBase.extend({
  fieldKind: z.literal('output'),
});
export type FieldTemplateBase = z.infer<typeof zFieldTemplateBase>;
export type FieldInputTemplateBase = z.infer<typeof zFieldInputTemplateBase>;
export type FieldOutputTemplateBase = z.infer<typeof zFieldOutputTemplateBase>;

export const zFieldTypeBase = z.object({
  isCollection: z.boolean(),
  isCollectionOrScalar: z.boolean(),
});

export const zFieldIdentifier = z.object({
  nodeId: z.string().trim().min(1),
  fieldName: z.string().trim().min(1),
});
export type FieldIdentifier = z.infer<typeof zFieldIdentifier>;
export const isFieldIdentifier = (val: unknown): val is FieldIdentifier => zFieldIdentifier.safeParse(val).success;
// #endregion

// #region IntegerField
export const zIntegerFieldType = zFieldTypeBase.extend({
  name: z.literal('IntegerField'),
});
export const zIntegerFieldValue = z.number().int();
export const zIntegerFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zIntegerFieldType,
  value: zIntegerFieldValue,
});
export const zIntegerFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zIntegerFieldType,
});
export const zIntegerFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zIntegerFieldType,
  default: zIntegerFieldValue,
  multipleOf: z.number().int().optional(),
  maximum: z.number().int().optional(),
  exclusiveMaximum: z.number().int().optional(),
  minimum: z.number().int().optional(),
  exclusiveMinimum: z.number().int().optional(),
});
export const zIntegerFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zIntegerFieldType,
});
export type IntegerFieldType = z.infer<typeof zIntegerFieldType>;
export type IntegerFieldValue = z.infer<typeof zIntegerFieldValue>;
export type IntegerFieldInputInstance = z.infer<typeof zIntegerFieldInputInstance>;
export type IntegerFieldInputTemplate = z.infer<typeof zIntegerFieldInputTemplate>;
export const isIntegerFieldInputInstance = (val: unknown): val is IntegerFieldInputInstance =>
  zIntegerFieldInputInstance.safeParse(val).success;
export const isIntegerFieldInputTemplate = (val: unknown): val is IntegerFieldInputTemplate =>
  zIntegerFieldInputTemplate.safeParse(val).success;
// #endregion

// #region FloatField
export const zFloatFieldType = zFieldTypeBase.extend({
  name: z.literal('FloatField'),
});
export const zFloatFieldValue = z.number();
export const zFloatFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zFloatFieldType,
  value: zFloatFieldValue,
});
export const zFloatFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zFloatFieldType,
});
export const zFloatFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zFloatFieldType,
  default: zFloatFieldValue,
  multipleOf: z.number().optional(),
  maximum: z.number().optional(),
  exclusiveMaximum: z.number().optional(),
  minimum: z.number().optional(),
  exclusiveMinimum: z.number().optional(),
});
export const zFloatFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zFloatFieldType,
});
export type FloatFieldType = z.infer<typeof zFloatFieldType>;
export type FloatFieldValue = z.infer<typeof zFloatFieldValue>;
export type FloatFieldInputInstance = z.infer<typeof zFloatFieldInputInstance>;
export type FloatFieldOutputInstance = z.infer<typeof zFloatFieldOutputInstance>;
export type FloatFieldInputTemplate = z.infer<typeof zFloatFieldInputTemplate>;
export type FloatFieldOutputTemplate = z.infer<typeof zFloatFieldOutputTemplate>;
export const isFloatFieldInputInstance = (val: unknown): val is FloatFieldInputInstance =>
  zFloatFieldInputInstance.safeParse(val).success;
export const isFloatFieldInputTemplate = (val: unknown): val is FloatFieldInputTemplate =>
  zFloatFieldInputTemplate.safeParse(val).success;
// #endregion

// #region StringField
export const zStringFieldType = zFieldTypeBase.extend({
  name: z.literal('StringField'),
});
export const zStringFieldValue = z.string();
export const zStringFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zStringFieldType,
  value: zStringFieldValue,
});
export const zStringFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zStringFieldType,
});
export const zStringFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zStringFieldType,
  default: zStringFieldValue,
  maxLength: z.number().int().optional(),
  minLength: z.number().int().optional(),
});
export const zStringFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zStringFieldType,
});

export type StringFieldType = z.infer<typeof zStringFieldType>;
export type StringFieldValue = z.infer<typeof zStringFieldValue>;
export type StringFieldInputInstance = z.infer<typeof zStringFieldInputInstance>;
export type StringFieldOutputInstance = z.infer<typeof zStringFieldOutputInstance>;
export type StringFieldInputTemplate = z.infer<typeof zStringFieldInputTemplate>;
export type StringFieldOutputTemplate = z.infer<typeof zStringFieldOutputTemplate>;
export const isStringFieldInputInstance = (val: unknown): val is StringFieldInputInstance =>
  zStringFieldInputInstance.safeParse(val).success;
export const isStringFieldInputTemplate = (val: unknown): val is StringFieldInputTemplate =>
  zStringFieldInputTemplate.safeParse(val).success;
// #endregion

// #region BooleanField
export const zBooleanFieldType = zFieldTypeBase.extend({
  name: z.literal('BooleanField'),
});
export const zBooleanFieldValue = z.boolean();
export const zBooleanFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zBooleanFieldType,
  value: zBooleanFieldValue,
});
export const zBooleanFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zBooleanFieldType,
});
export const zBooleanFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zBooleanFieldType,
  default: zBooleanFieldValue,
});
export const zBooleanFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zBooleanFieldType,
});
export type BooleanFieldType = z.infer<typeof zBooleanFieldType>;
export type BooleanFieldValue = z.infer<typeof zBooleanFieldValue>;
export type BooleanFieldInputInstance = z.infer<typeof zBooleanFieldInputInstance>;
export type BooleanFieldOutputInstance = z.infer<typeof zBooleanFieldOutputInstance>;
export type BooleanFieldInputTemplate = z.infer<typeof zBooleanFieldInputTemplate>;
export type BooleanFieldOutputTemplate = z.infer<typeof zBooleanFieldOutputTemplate>;
export const isBooleanFieldInputInstance = (val: unknown): val is BooleanFieldInputInstance =>
  zBooleanFieldInputInstance.safeParse(val).success;
export const isBooleanFieldInputTemplate = (val: unknown): val is BooleanFieldInputTemplate =>
  zBooleanFieldInputTemplate.safeParse(val).success;
// #endregion

// #region EnumField
export const zEnumFieldType = zFieldTypeBase.extend({
  name: z.literal('EnumField'),
});
export const zEnumFieldValue = z.string();
export const zEnumFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zEnumFieldType,
  value: zEnumFieldValue,
});
export const zEnumFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zEnumFieldType,
});
export const zEnumFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zEnumFieldType,
  default: zEnumFieldValue,
  options: z.array(z.string()),
  labels: z.record(z.string()).optional(),
});
export const zEnumFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zEnumFieldType,
});
export type EnumFieldType = z.infer<typeof zEnumFieldType>;
export type EnumFieldValue = z.infer<typeof zEnumFieldValue>;
export type EnumFieldInputInstance = z.infer<typeof zEnumFieldInputInstance>;
export type EnumFieldOutputInstance = z.infer<typeof zEnumFieldOutputInstance>;
export type EnumFieldInputTemplate = z.infer<typeof zEnumFieldInputTemplate>;
export type EnumFieldOutputTemplate = z.infer<typeof zEnumFieldOutputTemplate>;
export const isEnumFieldInputInstance = (val: unknown): val is EnumFieldInputInstance =>
  zEnumFieldInputInstance.safeParse(val).success;
export const isEnumFieldInputTemplate = (val: unknown): val is EnumFieldInputTemplate =>
  zEnumFieldInputTemplate.safeParse(val).success;
// #endregion

// #region ImageField
export const zImageFieldType = zFieldTypeBase.extend({
  name: z.literal('ImageField'),
});
export const zImageFieldValue = zImageField.optional();
export const zImageFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zImageFieldType,
  value: zImageFieldValue,
});
export const zImageFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zImageFieldType,
});
export const zImageFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zImageFieldType,
  default: zImageFieldValue,
});
export const zImageFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zImageFieldType,
});
export type ImageFieldType = z.infer<typeof zImageFieldType>;
export type ImageFieldValue = z.infer<typeof zImageFieldValue>;
export type ImageFieldInputInstance = z.infer<typeof zImageFieldInputInstance>;
export type ImageFieldOutputInstance = z.infer<typeof zImageFieldOutputInstance>;
export type ImageFieldInputTemplate = z.infer<typeof zImageFieldInputTemplate>;
export type ImageFieldOutputTemplate = z.infer<typeof zImageFieldOutputTemplate>;
export const isImageFieldInputInstance = (val: unknown): val is ImageFieldInputInstance =>
  zImageFieldInputInstance.safeParse(val).success;
export const isImageFieldInputTemplate = (val: unknown): val is ImageFieldInputTemplate =>
  zImageFieldInputTemplate.safeParse(val).success;
// #endregion

// #region BoardField
export const zBoardFieldType = zFieldTypeBase.extend({
  name: z.literal('BoardField'),
});
export const zBoardFieldValue = zBoardField.optional();
export const zBoardFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zBoardFieldType,
  value: zBoardFieldValue,
});
export const zBoardFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zBoardFieldType,
});
export const zBoardFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zBoardFieldType,
  default: zBoardFieldValue,
});
export const zBoardFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zBoardFieldType,
});
export type BoardFieldType = z.infer<typeof zBoardFieldType>;
export type BoardFieldValue = z.infer<typeof zBoardFieldValue>;
export type BoardFieldInputInstance = z.infer<typeof zBoardFieldInputInstance>;
export type BoardFieldOutputInstance = z.infer<typeof zBoardFieldOutputInstance>;
export type BoardFieldInputTemplate = z.infer<typeof zBoardFieldInputTemplate>;
export type BoardFieldOutputTemplate = z.infer<typeof zBoardFieldOutputTemplate>;
export const isBoardFieldInputInstance = (val: unknown): val is BoardFieldInputInstance =>
  zBoardFieldInputInstance.safeParse(val).success;
export const isBoardFieldInputTemplate = (val: unknown): val is BoardFieldInputTemplate =>
  zBoardFieldInputTemplate.safeParse(val).success;
// #endregion

// #region ColorField
export const zColorFieldType = zFieldTypeBase.extend({
  name: z.literal('ColorField'),
});
export const zColorFieldValue = zColorField.optional();
export const zColorFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zColorFieldType,
  value: zColorFieldValue,
});
export const zColorFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zColorFieldType,
});
export const zColorFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zColorFieldType,
  default: zColorFieldValue,
});
export const zColorFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zColorFieldType,
});
export type ColorFieldType = z.infer<typeof zColorFieldType>;
export type ColorFieldValue = z.infer<typeof zColorFieldValue>;
export type ColorFieldInputInstance = z.infer<typeof zColorFieldInputInstance>;
export type ColorFieldOutputInstance = z.infer<typeof zColorFieldOutputInstance>;
export type ColorFieldInputTemplate = z.infer<typeof zColorFieldInputTemplate>;
export type ColorFieldOutputTemplate = z.infer<typeof zColorFieldOutputTemplate>;
export const isColorFieldInputInstance = (val: unknown): val is ColorFieldInputInstance =>
  zColorFieldInputInstance.safeParse(val).success;
export const isColorFieldInputTemplate = (val: unknown): val is ColorFieldInputTemplate =>
  zColorFieldInputTemplate.safeParse(val).success;
// #endregion

// #region MainModelField
export const zMainModelFieldType = zFieldTypeBase.extend({
  name: z.literal('MainModelField'),
});
export const zMainModelFieldValue = zMainModelField.optional();
export const zMainModelFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zMainModelFieldType,
  value: zMainModelFieldValue,
});
export const zMainModelFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zMainModelFieldType,
});
export const zMainModelFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zMainModelFieldType,
  default: zMainModelFieldValue,
});
export const zMainModelFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zMainModelFieldType,
});
export type MainModelFieldType = z.infer<typeof zMainModelFieldType>;
export type MainModelFieldValue = z.infer<typeof zMainModelFieldValue>;
export type MainModelFieldInputInstance = z.infer<typeof zMainModelFieldInputInstance>;
export type MainModelFieldOutputInstance = z.infer<typeof zMainModelFieldOutputInstance>;
export type MainModelFieldInputTemplate = z.infer<typeof zMainModelFieldInputTemplate>;
export type MainModelFieldOutputTemplate = z.infer<typeof zMainModelFieldOutputTemplate>;
export const isMainModelFieldInputInstance = (val: unknown): val is MainModelFieldInputInstance =>
  zMainModelFieldInputInstance.safeParse(val).success;
export const isMainModelFieldInputTemplate = (val: unknown): val is MainModelFieldInputTemplate =>
  zMainModelFieldInputTemplate.safeParse(val).success;
// #endregion

// #region SDXLMainModelField
export const zSDXLMainModelFieldType = zFieldTypeBase.extend({
  name: z.literal('SDXLMainModelField'),
});
export const zSDXLMainModelFieldValue = zMainModelFieldValue; // TODO: Narrow to SDXL models only.
export const zSDXLMainModelFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zSDXLMainModelFieldType,
  value: zSDXLMainModelFieldValue,
});
export const zSDXLMainModelFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zSDXLMainModelFieldType,
});
export const zSDXLMainModelFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zSDXLMainModelFieldType,
  default: zSDXLMainModelFieldValue,
});
export const zSDXLMainModelFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zSDXLMainModelFieldType,
});
export type SDXLMainModelFieldType = z.infer<typeof zSDXLMainModelFieldType>;
export type SDXLMainModelFieldValue = z.infer<typeof zSDXLMainModelFieldValue>;
export type SDXLMainModelFieldInputInstance = z.infer<typeof zSDXLMainModelFieldInputInstance>;
export type SDXLMainModelFieldOutputInstance = z.infer<typeof zSDXLMainModelFieldOutputInstance>;
export type SDXLMainModelFieldInputTemplate = z.infer<typeof zSDXLMainModelFieldInputTemplate>;
export type SDXLMainModelFieldOutputTemplate = z.infer<typeof zSDXLMainModelFieldOutputTemplate>;
export const isSDXLMainModelFieldInputInstance = (val: unknown): val is SDXLMainModelFieldInputInstance =>
  zSDXLMainModelFieldInputInstance.safeParse(val).success;
export const isSDXLMainModelFieldInputTemplate = (val: unknown): val is SDXLMainModelFieldInputTemplate =>
  zSDXLMainModelFieldInputTemplate.safeParse(val).success;
// #endregion

// #region SDXLRefinerModelField
export const zSDXLRefinerModelFieldType = zFieldTypeBase.extend({
  name: z.literal('SDXLRefinerModelField'),
});
export const zSDXLRefinerModelFieldValue = zMainModelFieldValue; // TODO: Narrow to SDXL Refiner models only.
export const zSDXLRefinerModelFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zSDXLRefinerModelFieldType,
  value: zSDXLRefinerModelFieldValue,
});
export const zSDXLRefinerModelFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zSDXLRefinerModelFieldType,
});
export const zSDXLRefinerModelFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zSDXLRefinerModelFieldType,
  default: zSDXLRefinerModelFieldValue,
});
export const zSDXLRefinerModelFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zSDXLRefinerModelFieldType,
});
export type SDXLRefinerModelFieldType = z.infer<typeof zSDXLRefinerModelFieldType>;
export type SDXLRefinerModelFieldValue = z.infer<typeof zSDXLRefinerModelFieldValue>;
export type SDXLRefinerModelFieldInputInstance = z.infer<typeof zSDXLRefinerModelFieldInputInstance>;
export type SDXLRefinerModelFieldOutputInstance = z.infer<typeof zSDXLRefinerModelFieldOutputInstance>;
export type SDXLRefinerModelFieldInputTemplate = z.infer<typeof zSDXLRefinerModelFieldInputTemplate>;
export type SDXLRefinerModelFieldOutputTemplate = z.infer<typeof zSDXLRefinerModelFieldOutputTemplate>;
export const isSDXLRefinerModelFieldInputInstance = (val: unknown): val is SDXLRefinerModelFieldInputInstance =>
  zSDXLRefinerModelFieldInputInstance.safeParse(val).success;
export const isSDXLRefinerModelFieldInputTemplate = (val: unknown): val is SDXLRefinerModelFieldInputTemplate =>
  zSDXLRefinerModelFieldInputTemplate.safeParse(val).success;
// #endregion

// #region VAEModelField
export const zVAEModelFieldType = zFieldTypeBase.extend({
  name: z.literal('VAEModelField'),
});
export const zVAEModelFieldValue = zVAEModelField.optional();
export const zVAEModelFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zVAEModelFieldType,
  value: zVAEModelFieldValue,
});
export const zVAEModelFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zVAEModelFieldType,
});
export const zVAEModelFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zVAEModelFieldType,
  default: zVAEModelFieldValue,
});
export const zVAEModelFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zVAEModelFieldType,
});
export type VAEModelFieldType = z.infer<typeof zVAEModelFieldType>;
export type VAEModelFieldValue = z.infer<typeof zVAEModelFieldValue>;
export type VAEModelFieldInputInstance = z.infer<typeof zVAEModelFieldInputInstance>;
export type VAEModelFieldOutputInstance = z.infer<typeof zVAEModelFieldOutputInstance>;
export type VAEModelFieldInputTemplate = z.infer<typeof zVAEModelFieldInputTemplate>;
export type VAEModelFieldOutputTemplate = z.infer<typeof zVAEModelFieldOutputTemplate>;
export const isVAEModelFieldInputInstance = (val: unknown): val is VAEModelFieldInputInstance =>
  zVAEModelFieldInputInstance.safeParse(val).success;
export const isVAEModelFieldInputTemplate = (val: unknown): val is VAEModelFieldInputTemplate =>
  zVAEModelFieldInputTemplate.safeParse(val).success;
// #endregion

// #region LoRAModelField
export const zLoRAModelFieldType = zFieldTypeBase.extend({
  name: z.literal('LoRAModelField'),
});
export const zLoRAModelFieldValue = zLoRAModelField.optional();
export const zLoRAModelFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zLoRAModelFieldType,
  value: zLoRAModelFieldValue,
});
export const zLoRAModelFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zLoRAModelFieldType,
});
export const zLoRAModelFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zLoRAModelFieldType,
  default: zLoRAModelFieldValue,
});
export const zLoRAModelFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zLoRAModelFieldType,
});
export type LoRAModelFieldType = z.infer<typeof zLoRAModelFieldType>;
export type LoRAModelFieldValue = z.infer<typeof zLoRAModelFieldValue>;
export type LoRAModelFieldInputInstance = z.infer<typeof zLoRAModelFieldInputInstance>;
export type LoRAModelFieldOutputInstance = z.infer<typeof zLoRAModelFieldOutputInstance>;
export type LoRAModelFieldInputTemplate = z.infer<typeof zLoRAModelFieldInputTemplate>;
export type LoRAModelFieldOutputTemplate = z.infer<typeof zLoRAModelFieldOutputTemplate>;
export const isLoRAModelFieldInputInstance = (val: unknown): val is LoRAModelFieldInputInstance =>
  zLoRAModelFieldInputInstance.safeParse(val).success;
export const isLoRAModelFieldInputTemplate = (val: unknown): val is LoRAModelFieldInputTemplate =>
  zLoRAModelFieldInputTemplate.safeParse(val).success;
// #endregion

// #region ControlNetModelField
export const zControlNetModelFieldType = zFieldTypeBase.extend({
  name: z.literal('ControlNetModelField'),
});
export const zControlNetModelFieldValue = zControlNetModelField.optional();
export const zControlNetModelFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zControlNetModelFieldType,
  value: zControlNetModelFieldValue,
});
export const zControlNetModelFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zControlNetModelFieldType,
});
export const zControlNetModelFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zControlNetModelFieldType,
  default: zControlNetModelFieldValue,
});
export const zControlNetModelFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zControlNetModelFieldType,
});
export type ControlNetModelFieldType = z.infer<typeof zControlNetModelFieldType>;
export type ControlNetModelFieldValue = z.infer<typeof zControlNetModelFieldValue>;
export type ControlNetModelFieldInputInstance = z.infer<typeof zControlNetModelFieldInputInstance>;
export type ControlNetModelFieldOutputInstance = z.infer<typeof zControlNetModelFieldOutputInstance>;
export type ControlNetModelFieldInputTemplate = z.infer<typeof zControlNetModelFieldInputTemplate>;
export type ControlNetModelFieldOutputTemplate = z.infer<typeof zControlNetModelFieldOutputTemplate>;
export const isControlNetModelFieldInputInstance = (val: unknown): val is ControlNetModelFieldInputInstance =>
  zControlNetModelFieldInputInstance.safeParse(val).success;
export const isControlNetModelFieldInputTemplate = (val: unknown): val is ControlNetModelFieldInputTemplate =>
  zControlNetModelFieldInputTemplate.safeParse(val).success;
export const isControlNetModelFieldValue = (v: unknown): v is ControlNetModelFieldValue =>
  zControlNetModelFieldValue.safeParse(v).success;
// #endregion

// #region IPAdapterModelField
export const zIPAdapterModelFieldType = zFieldTypeBase.extend({
  name: z.literal('IPAdapterModelField'),
});
export const zIPAdapterModelFieldValue = zIPAdapterModelField.optional();
export const zIPAdapterModelFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zIPAdapterModelFieldType,
  value: zIPAdapterModelFieldValue,
});
export const zIPAdapterModelFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zIPAdapterModelFieldType,
});
export const zIPAdapterModelFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zIPAdapterModelFieldType,
  default: zIPAdapterModelFieldValue,
});
export const zIPAdapterModelFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zIPAdapterModelFieldType,
});
export type IPAdapterModelFieldType = z.infer<typeof zIPAdapterModelFieldType>;
export type IPAdapterModelFieldValue = z.infer<typeof zIPAdapterModelFieldValue>;
export type IPAdapterModelFieldInputInstance = z.infer<typeof zIPAdapterModelFieldInputInstance>;
export type IPAdapterModelFieldOutputInstance = z.infer<typeof zIPAdapterModelFieldOutputInstance>;
export type IPAdapterModelFieldInputTemplate = z.infer<typeof zIPAdapterModelFieldInputTemplate>;
export type IPAdapterModelFieldOutputTemplate = z.infer<typeof zIPAdapterModelFieldOutputTemplate>;
export const isIPAdapterModelFieldInputInstance = (val: unknown): val is IPAdapterModelFieldInputInstance =>
  zIPAdapterModelFieldInputInstance.safeParse(val).success;
export const isIPAdapterModelFieldInputTemplate = (val: unknown): val is IPAdapterModelFieldInputTemplate =>
  zIPAdapterModelFieldInputTemplate.safeParse(val).success;
export const isIPAdapterModelFieldValue = (val: unknown): val is IPAdapterModelFieldValue =>
  zIPAdapterModelFieldValue.safeParse(val).success;
// #endregion

// #region T2IAdapterField
export const zT2IAdapterModelFieldType = zFieldTypeBase.extend({
  name: z.literal('T2IAdapterModelField'),
});
export const zT2IAdapterModelFieldValue = zT2IAdapterModelField.optional();
export const zT2IAdapterModelFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zT2IAdapterModelFieldType,
  value: zT2IAdapterModelFieldValue,
});
export const zT2IAdapterModelFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zT2IAdapterModelFieldType,
});
export const zT2IAdapterModelFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zT2IAdapterModelFieldType,
  default: zT2IAdapterModelFieldValue,
});
export const zT2IAdapterModelFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zT2IAdapterModelFieldType,
});
export type T2IAdapterModelFieldType = z.infer<typeof zT2IAdapterModelFieldType>;
export type T2IAdapterModelFieldValue = z.infer<typeof zT2IAdapterModelFieldValue>;
export type T2IAdapterModelFieldInputInstance = z.infer<typeof zT2IAdapterModelFieldInputInstance>;
export type T2IAdapterModelFieldOutputInstance = z.infer<typeof zT2IAdapterModelFieldOutputInstance>;
export type T2IAdapterModelFieldInputTemplate = z.infer<typeof zT2IAdapterModelFieldInputTemplate>;
export type T2IAdapterModelFieldOutputTemplate = z.infer<typeof zT2IAdapterModelFieldOutputTemplate>;
export const isT2IAdapterModelFieldInputInstance = (val: unknown): val is T2IAdapterModelFieldInputInstance =>
  zT2IAdapterModelFieldInputInstance.safeParse(val).success;
export const isT2IAdapterModelFieldInputTemplate = (val: unknown): val is T2IAdapterModelFieldInputTemplate =>
  zT2IAdapterModelFieldInputTemplate.safeParse(val).success;
// #endregion

// #region SchedulerField
export const zSchedulerFieldType = zFieldTypeBase.extend({
  name: z.literal('SchedulerField'),
});
export const zSchedulerFieldValue = zSchedulerField.optional();
export const zSchedulerFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zSchedulerFieldType,
  value: zSchedulerFieldValue,
});
export const zSchedulerFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zSchedulerFieldType,
});
export const zSchedulerFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zSchedulerFieldType,
  default: zSchedulerFieldValue,
});
export const zSchedulerFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zSchedulerFieldType,
});
export type SchedulerFieldType = z.infer<typeof zSchedulerFieldType>;
export type SchedulerFieldValue = z.infer<typeof zSchedulerFieldValue>;
export type SchedulerFieldInputInstance = z.infer<typeof zSchedulerFieldInputInstance>;
export type SchedulerFieldOutputInstance = z.infer<typeof zSchedulerFieldOutputInstance>;
export type SchedulerFieldInputTemplate = z.infer<typeof zSchedulerFieldInputTemplate>;
export type SchedulerFieldOutputTemplate = z.infer<typeof zSchedulerFieldOutputTemplate>;
export const isSchedulerFieldInputInstance = (val: unknown): val is SchedulerFieldInputInstance =>
  zSchedulerFieldInputInstance.safeParse(val).success;
export const isSchedulerFieldInputTemplate = (val: unknown): val is SchedulerFieldInputTemplate =>
  zSchedulerFieldInputTemplate.safeParse(val).success;
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
export const zStatelessFieldType = zFieldTypeBase.extend({
  name: z.string().min(1), // stateless --> we accept the field's name as the type
});
export const zStatelessFieldValue = z.undefined().catch(undefined); // stateless --> no value, but making this z.never() introduces a lot of extra TS fanagling
export const zStatelessFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zStatelessFieldType,
  value: zStatelessFieldValue,
});
export const zStatelessFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zStatelessFieldType,
});
export const zStatelessFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zStatelessFieldType,
  default: zStatelessFieldValue,
  input: z.literal('connection'), // stateless --> only accepts connection inputs
});
export const zStatelessFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zStatelessFieldType,
});

export type StatelessFieldType = z.infer<typeof zStatelessFieldType>;
export type StatelessFieldValue = z.infer<typeof zStatelessFieldValue>;
export type StatelessFieldInputInstance = z.infer<typeof zStatelessFieldInputInstance>;
export type StatelessFieldOutputInstance = z.infer<typeof zStatelessFieldOutputInstance>;
export type StatelessFieldInputTemplate = z.infer<typeof zStatelessFieldInputTemplate>;
export type StatelessFieldOutputTemplate = z.infer<typeof zStatelessFieldOutputTemplate>;
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

// #region StatefulFieldType & FieldType
export const zStatefulFieldType = z.union([
  zIntegerFieldType,
  zFloatFieldType,
  zStringFieldType,
  zBooleanFieldType,
  zEnumFieldType,
  zImageFieldType,
  zBoardFieldType,
  zMainModelFieldType,
  zSDXLMainModelFieldType,
  zSDXLRefinerModelFieldType,
  zVAEModelFieldType,
  zLoRAModelFieldType,
  zControlNetModelFieldType,
  zIPAdapterModelFieldType,
  zT2IAdapterModelFieldType,
  zColorFieldType,
  zSchedulerFieldType,
]);
export type StatefulFieldType = z.infer<typeof zStatefulFieldType>;
export const isStatefulFieldType = (val: unknown): val is StatefulFieldType =>
  zStatefulFieldType.safeParse(val).success;

export const zFieldType = z.union([zStatefulFieldType, zStatelessFieldType]);
export type FieldType = z.infer<typeof zFieldType>;
export const isFieldType = (val: unknown): val is FieldType => zFieldType.safeParse(val).success;
// #endregion

// #region StatefulFieldValue & FieldValue
export const zStatefulFieldValue = z.union([
  zIntegerFieldValue,
  zFloatFieldValue,
  zStringFieldValue,
  zBooleanFieldValue,
  zEnumFieldValue,
  zImageFieldValue,
  zBoardFieldValue,
  zMainModelFieldValue,
  zSDXLMainModelFieldValue,
  zSDXLRefinerModelFieldValue,
  zVAEModelFieldValue,
  zLoRAModelFieldValue,
  zControlNetModelFieldValue,
  zIPAdapterModelFieldValue,
  zT2IAdapterModelFieldValue,
  zColorFieldValue,
  zSchedulerFieldValue,
]);
export type StatefulFieldValue = z.infer<typeof zStatefulFieldValue>;
export const isStatefulFieldValue = (val: unknown): val is StatefulFieldValue =>
  zStatefulFieldValue.safeParse(val).success;

export const zFieldValue = z.union([zStatefulFieldValue, zStatelessFieldValue]);
export type FieldValue = z.infer<typeof zFieldValue>;
export const isFieldValue = (val: unknown): val is FieldValue => zFieldValue.safeParse(val).success;
// #endregion

// #region StatefulFieldInputInstance & FieldInputInstance
export const zStatefulFieldInputInstance = z.union([
  zIntegerFieldInputInstance,
  zFloatFieldInputInstance,
  zStringFieldInputInstance,
  zBooleanFieldInputInstance,
  zEnumFieldInputInstance,
  zImageFieldInputInstance,
  zBoardFieldInputInstance,
  zMainModelFieldInputInstance,
  zSDXLMainModelFieldInputInstance,
  zSDXLRefinerModelFieldInputInstance,
  zVAEModelFieldInputInstance,
  zLoRAModelFieldInputInstance,
  zControlNetModelFieldInputInstance,
  zIPAdapterModelFieldInputInstance,
  zT2IAdapterModelFieldInputInstance,
  zColorFieldInputInstance,
  zSchedulerFieldInputInstance,
]);
export type StatefulFieldInputInstance = z.infer<typeof zStatefulFieldInputInstance>;
export const isStatefulFieldInputInstance = (val: unknown): val is StatefulFieldInputInstance =>
  zStatefulFieldInputInstance.safeParse(val).success;

export const zFieldInputInstance = z.union([zStatefulFieldInputInstance, zStatelessFieldInputInstance]);
export type FieldInputInstance = z.infer<typeof zFieldInputInstance>;
export const isFieldInputInstance = (val: unknown): val is FieldInputInstance =>
  zFieldInputInstance.safeParse(val).success;
// #endregion

// #region StatefulFieldOutputInstance & FieldOutputInstance
export const zStatefulFieldOutputInstance = z.union([
  zIntegerFieldOutputInstance,
  zFloatFieldOutputInstance,
  zStringFieldOutputInstance,
  zBooleanFieldOutputInstance,
  zEnumFieldOutputInstance,
  zImageFieldOutputInstance,
  zBoardFieldOutputInstance,
  zMainModelFieldOutputInstance,
  zSDXLMainModelFieldOutputInstance,
  zSDXLRefinerModelFieldOutputInstance,
  zVAEModelFieldOutputInstance,
  zLoRAModelFieldOutputInstance,
  zControlNetModelFieldOutputInstance,
  zIPAdapterModelFieldOutputInstance,
  zT2IAdapterModelFieldOutputInstance,
  zColorFieldOutputInstance,
  zSchedulerFieldOutputInstance,
]);
export type StatefulFieldOutputInstance = z.infer<typeof zStatefulFieldOutputInstance>;
export const isStatefulFieldOutputInstance = (val: unknown): val is StatefulFieldOutputInstance =>
  zStatefulFieldOutputInstance.safeParse(val).success;

export const zFieldOutputInstance = z.union([zStatefulFieldOutputInstance, zStatelessFieldOutputInstance]);
export type FieldOutputInstance = z.infer<typeof zFieldOutputInstance>;
export const isFieldOutputInstance = (val: unknown): val is FieldOutputInstance =>
  zFieldOutputInstance.safeParse(val).success;
// #endregion

// #region StatefulFieldInputTemplate & FieldInputTemplate
export const zStatefulFieldInputTemplate = z.union([
  zIntegerFieldInputTemplate,
  zFloatFieldInputTemplate,
  zStringFieldInputTemplate,
  zBooleanFieldInputTemplate,
  zEnumFieldInputTemplate,
  zImageFieldInputTemplate,
  zBoardFieldInputTemplate,
  zMainModelFieldInputTemplate,
  zSDXLMainModelFieldInputTemplate,
  zSDXLRefinerModelFieldInputTemplate,
  zVAEModelFieldInputTemplate,
  zLoRAModelFieldInputTemplate,
  zControlNetModelFieldInputTemplate,
  zIPAdapterModelFieldInputTemplate,
  zT2IAdapterModelFieldInputTemplate,
  zColorFieldInputTemplate,
  zSchedulerFieldInputTemplate,
  zStatelessFieldInputTemplate,
]);
export type StatefulFieldInputTemplate = z.infer<typeof zFieldInputTemplate>;
export const isStatefulFieldInputTemplate = (val: unknown): val is StatefulFieldInputTemplate =>
  zStatefulFieldInputTemplate.safeParse(val).success;

export const zFieldInputTemplate = z.union([zStatefulFieldInputTemplate, zStatelessFieldInputTemplate]);
export type FieldInputTemplate = z.infer<typeof zFieldInputTemplate>;
export const isFieldInputTemplate = (val: unknown): val is FieldInputTemplate =>
  zFieldInputTemplate.safeParse(val).success;
// #endregion

// #region StatefulFieldOutputTemplate & FieldOutputTemplate
export const zStatefulFieldOutputTemplate = z.union([
  zIntegerFieldOutputTemplate,
  zFloatFieldOutputTemplate,
  zStringFieldOutputTemplate,
  zBooleanFieldOutputTemplate,
  zEnumFieldOutputTemplate,
  zImageFieldOutputTemplate,
  zBoardFieldOutputTemplate,
  zMainModelFieldOutputTemplate,
  zSDXLMainModelFieldOutputTemplate,
  zSDXLRefinerModelFieldOutputTemplate,
  zVAEModelFieldOutputTemplate,
  zLoRAModelFieldOutputTemplate,
  zControlNetModelFieldOutputTemplate,
  zIPAdapterModelFieldOutputTemplate,
  zT2IAdapterModelFieldOutputTemplate,
  zColorFieldOutputTemplate,
  zSchedulerFieldOutputTemplate,
]);
export type StatefulFieldOutputTemplate = z.infer<typeof zStatefulFieldOutputTemplate>;
export const isStatefulFieldOutputTemplate = (val: unknown): val is StatefulFieldOutputTemplate =>
  zStatefulFieldOutputTemplate.safeParse(val).success;

export const zFieldOutputTemplate = z.union([zStatefulFieldOutputTemplate, zStatelessFieldOutputTemplate]);
export type FieldOutputTemplate = z.infer<typeof zFieldOutputTemplate>;
export const isFieldOutputTemplate = (val: unknown): val is FieldOutputTemplate =>
  zFieldOutputTemplate.safeParse(val).success;
// #endregion
