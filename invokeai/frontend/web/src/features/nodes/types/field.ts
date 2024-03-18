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
  label: z.string().nullish(),
});
const zFieldTemplateBase = z.object({
  name: z.string().min(1),
  title: z.string().min(1),
  description: z.string().nullish(),
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

const zFieldTypeBase = z.object({
  isCollection: z.boolean(),
  isCollectionOrScalar: z.boolean(),
});

export const zFieldIdentifier = z.object({
  nodeId: z.string().trim().min(1),
  fieldName: z.string().trim().min(1),
});
export type FieldIdentifier = z.infer<typeof zFieldIdentifier>;
// #endregion

// #region IntegerField
const zIntegerFieldType = zFieldTypeBase.extend({
  name: z.literal('IntegerField'),
});
export const zIntegerFieldValue = z.number().int();
const zIntegerFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zIntegerFieldValue,
});
const zIntegerFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zIntegerFieldType,
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
export const isIntegerFieldInputInstance = (val: unknown): val is IntegerFieldInputInstance =>
  zIntegerFieldInputInstance.safeParse(val).success;
export const isIntegerFieldInputTemplate = (val: unknown): val is IntegerFieldInputTemplate =>
  zIntegerFieldInputTemplate.safeParse(val).success;
// #endregion

// #region FloatField
const zFloatFieldType = zFieldTypeBase.extend({
  name: z.literal('FloatField'),
});
export const zFloatFieldValue = z.number();
const zFloatFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zFloatFieldValue,
});
const zFloatFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zFloatFieldType,
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
export const isFloatFieldInputInstance = (val: unknown): val is FloatFieldInputInstance =>
  zFloatFieldInputInstance.safeParse(val).success;
export const isFloatFieldInputTemplate = (val: unknown): val is FloatFieldInputTemplate =>
  zFloatFieldInputTemplate.safeParse(val).success;
// #endregion

// #region StringField
const zStringFieldType = zFieldTypeBase.extend({
  name: z.literal('StringField'),
});
export const zStringFieldValue = z.string();
const zStringFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zStringFieldValue,
});
const zStringFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zStringFieldType,
  default: zStringFieldValue,
  maxLength: z.number().int().optional(),
  minLength: z.number().int().optional(),
});
const zStringFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zStringFieldType,
});

export type StringFieldValue = z.infer<typeof zStringFieldValue>;
export type StringFieldInputInstance = z.infer<typeof zStringFieldInputInstance>;
export type StringFieldInputTemplate = z.infer<typeof zStringFieldInputTemplate>;
export const isStringFieldInputInstance = (val: unknown): val is StringFieldInputInstance =>
  zStringFieldInputInstance.safeParse(val).success;
export const isStringFieldInputTemplate = (val: unknown): val is StringFieldInputTemplate =>
  zStringFieldInputTemplate.safeParse(val).success;
// #endregion

// #region BooleanField
const zBooleanFieldType = zFieldTypeBase.extend({
  name: z.literal('BooleanField'),
});
export const zBooleanFieldValue = z.boolean();
const zBooleanFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zBooleanFieldValue,
});
const zBooleanFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zBooleanFieldType,
  default: zBooleanFieldValue,
});
const zBooleanFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zBooleanFieldType,
});
export type BooleanFieldValue = z.infer<typeof zBooleanFieldValue>;
export type BooleanFieldInputInstance = z.infer<typeof zBooleanFieldInputInstance>;
export type BooleanFieldInputTemplate = z.infer<typeof zBooleanFieldInputTemplate>;
export const isBooleanFieldInputInstance = (val: unknown): val is BooleanFieldInputInstance =>
  zBooleanFieldInputInstance.safeParse(val).success;
export const isBooleanFieldInputTemplate = (val: unknown): val is BooleanFieldInputTemplate =>
  zBooleanFieldInputTemplate.safeParse(val).success;
// #endregion

// #region EnumField
const zEnumFieldType = zFieldTypeBase.extend({
  name: z.literal('EnumField'),
});
export const zEnumFieldValue = z.string();
const zEnumFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zEnumFieldValue,
});
const zEnumFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zEnumFieldType,
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
export const isEnumFieldInputInstance = (val: unknown): val is EnumFieldInputInstance =>
  zEnumFieldInputInstance.safeParse(val).success;
export const isEnumFieldInputTemplate = (val: unknown): val is EnumFieldInputTemplate =>
  zEnumFieldInputTemplate.safeParse(val).success;
// #endregion

// #region ImageField
const zImageFieldType = zFieldTypeBase.extend({
  name: z.literal('ImageField'),
});
export const zImageFieldValue = zImageField.optional();
const zImageFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zImageFieldValue,
});
const zImageFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zImageFieldType,
  default: zImageFieldValue,
});
const zImageFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zImageFieldType,
});
export type ImageFieldValue = z.infer<typeof zImageFieldValue>;
export type ImageFieldInputInstance = z.infer<typeof zImageFieldInputInstance>;
export type ImageFieldInputTemplate = z.infer<typeof zImageFieldInputTemplate>;
export const isImageFieldInputInstance = (val: unknown): val is ImageFieldInputInstance =>
  zImageFieldInputInstance.safeParse(val).success;
export const isImageFieldInputTemplate = (val: unknown): val is ImageFieldInputTemplate =>
  zImageFieldInputTemplate.safeParse(val).success;
// #endregion

// #region BoardField
const zBoardFieldType = zFieldTypeBase.extend({
  name: z.literal('BoardField'),
});
export const zBoardFieldValue = zBoardField.optional();
const zBoardFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zBoardFieldValue,
});
const zBoardFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zBoardFieldType,
  default: zBoardFieldValue,
});
const zBoardFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zBoardFieldType,
});
export type BoardFieldValue = z.infer<typeof zBoardFieldValue>;
export type BoardFieldInputInstance = z.infer<typeof zBoardFieldInputInstance>;
export type BoardFieldInputTemplate = z.infer<typeof zBoardFieldInputTemplate>;
export const isBoardFieldInputInstance = (val: unknown): val is BoardFieldInputInstance =>
  zBoardFieldInputInstance.safeParse(val).success;
export const isBoardFieldInputTemplate = (val: unknown): val is BoardFieldInputTemplate =>
  zBoardFieldInputTemplate.safeParse(val).success;
// #endregion

// #region ColorField
const zColorFieldType = zFieldTypeBase.extend({
  name: z.literal('ColorField'),
});
export const zColorFieldValue = zColorField.optional();
const zColorFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zColorFieldValue,
});
const zColorFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zColorFieldType,
  default: zColorFieldValue,
});
const zColorFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zColorFieldType,
});
export type ColorFieldValue = z.infer<typeof zColorFieldValue>;
export type ColorFieldInputInstance = z.infer<typeof zColorFieldInputInstance>;
export type ColorFieldInputTemplate = z.infer<typeof zColorFieldInputTemplate>;
export const isColorFieldInputInstance = (val: unknown): val is ColorFieldInputInstance =>
  zColorFieldInputInstance.safeParse(val).success;
export const isColorFieldInputTemplate = (val: unknown): val is ColorFieldInputTemplate =>
  zColorFieldInputTemplate.safeParse(val).success;
// #endregion

// #region MainModelField
const zMainModelFieldType = zFieldTypeBase.extend({
  name: z.literal('MainModelField'),
});
export const zMainModelFieldValue = zModelIdentifierField.optional();
const zMainModelFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zMainModelFieldValue,
});
const zMainModelFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zMainModelFieldType,
  default: zMainModelFieldValue,
});
const zMainModelFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zMainModelFieldType,
});
export type MainModelFieldValue = z.infer<typeof zMainModelFieldValue>;
export type MainModelFieldInputInstance = z.infer<typeof zMainModelFieldInputInstance>;
export type MainModelFieldInputTemplate = z.infer<typeof zMainModelFieldInputTemplate>;
export const isMainModelFieldInputInstance = (val: unknown): val is MainModelFieldInputInstance =>
  zMainModelFieldInputInstance.safeParse(val).success;
export const isMainModelFieldInputTemplate = (val: unknown): val is MainModelFieldInputTemplate =>
  zMainModelFieldInputTemplate.safeParse(val).success;
// #endregion

// #region SDXLMainModelField
const zSDXLMainModelFieldType = zFieldTypeBase.extend({
  name: z.literal('SDXLMainModelField'),
});
const zSDXLMainModelFieldValue = zMainModelFieldValue; // TODO: Narrow to SDXL models only.
const zSDXLMainModelFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zSDXLMainModelFieldValue,
});
const zSDXLMainModelFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zSDXLMainModelFieldType,
  default: zSDXLMainModelFieldValue,
});
const zSDXLMainModelFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zSDXLMainModelFieldType,
});
export type SDXLMainModelFieldInputInstance = z.infer<typeof zSDXLMainModelFieldInputInstance>;
export type SDXLMainModelFieldInputTemplate = z.infer<typeof zSDXLMainModelFieldInputTemplate>;
export const isSDXLMainModelFieldInputInstance = (val: unknown): val is SDXLMainModelFieldInputInstance =>
  zSDXLMainModelFieldInputInstance.safeParse(val).success;
export const isSDXLMainModelFieldInputTemplate = (val: unknown): val is SDXLMainModelFieldInputTemplate =>
  zSDXLMainModelFieldInputTemplate.safeParse(val).success;
// #endregion

// #region SDXLRefinerModelField
const zSDXLRefinerModelFieldType = zFieldTypeBase.extend({
  name: z.literal('SDXLRefinerModelField'),
});
export const zSDXLRefinerModelFieldValue = zMainModelFieldValue; // TODO: Narrow to SDXL Refiner models only.
const zSDXLRefinerModelFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zSDXLRefinerModelFieldValue,
});
const zSDXLRefinerModelFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zSDXLRefinerModelFieldType,
  default: zSDXLRefinerModelFieldValue,
});
const zSDXLRefinerModelFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zSDXLRefinerModelFieldType,
});
export type SDXLRefinerModelFieldValue = z.infer<typeof zSDXLRefinerModelFieldValue>;
export type SDXLRefinerModelFieldInputInstance = z.infer<typeof zSDXLRefinerModelFieldInputInstance>;
export type SDXLRefinerModelFieldInputTemplate = z.infer<typeof zSDXLRefinerModelFieldInputTemplate>;
export const isSDXLRefinerModelFieldInputInstance = (val: unknown): val is SDXLRefinerModelFieldInputInstance =>
  zSDXLRefinerModelFieldInputInstance.safeParse(val).success;
export const isSDXLRefinerModelFieldInputTemplate = (val: unknown): val is SDXLRefinerModelFieldInputTemplate =>
  zSDXLRefinerModelFieldInputTemplate.safeParse(val).success;
// #endregion

// #region VAEModelField
const zVAEModelFieldType = zFieldTypeBase.extend({
  name: z.literal('VAEModelField'),
});
export const zVAEModelFieldValue = zModelIdentifierField.optional();
const zVAEModelFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zVAEModelFieldValue,
});
const zVAEModelFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zVAEModelFieldType,
  default: zVAEModelFieldValue,
});
const zVAEModelFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zVAEModelFieldType,
});
export type VAEModelFieldValue = z.infer<typeof zVAEModelFieldValue>;
export type VAEModelFieldInputInstance = z.infer<typeof zVAEModelFieldInputInstance>;
export type VAEModelFieldInputTemplate = z.infer<typeof zVAEModelFieldInputTemplate>;
export const isVAEModelFieldInputInstance = (val: unknown): val is VAEModelFieldInputInstance =>
  zVAEModelFieldInputInstance.safeParse(val).success;
export const isVAEModelFieldInputTemplate = (val: unknown): val is VAEModelFieldInputTemplate =>
  zVAEModelFieldInputTemplate.safeParse(val).success;
// #endregion

// #region LoRAModelField
const zLoRAModelFieldType = zFieldTypeBase.extend({
  name: z.literal('LoRAModelField'),
});
export const zLoRAModelFieldValue = zModelIdentifierField.optional();
const zLoRAModelFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zLoRAModelFieldValue,
});
const zLoRAModelFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zLoRAModelFieldType,
  default: zLoRAModelFieldValue,
});
const zLoRAModelFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zLoRAModelFieldType,
});
export type LoRAModelFieldValue = z.infer<typeof zLoRAModelFieldValue>;
export type LoRAModelFieldInputInstance = z.infer<typeof zLoRAModelFieldInputInstance>;
export type LoRAModelFieldInputTemplate = z.infer<typeof zLoRAModelFieldInputTemplate>;
export const isLoRAModelFieldInputInstance = (val: unknown): val is LoRAModelFieldInputInstance =>
  zLoRAModelFieldInputInstance.safeParse(val).success;
export const isLoRAModelFieldInputTemplate = (val: unknown): val is LoRAModelFieldInputTemplate =>
  zLoRAModelFieldInputTemplate.safeParse(val).success;
// #endregion

// #region ControlNetModelField
const zControlNetModelFieldType = zFieldTypeBase.extend({
  name: z.literal('ControlNetModelField'),
});
export const zControlNetModelFieldValue = zModelIdentifierField.optional();
const zControlNetModelFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zControlNetModelFieldValue,
});
const zControlNetModelFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zControlNetModelFieldType,
  default: zControlNetModelFieldValue,
});
const zControlNetModelFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zControlNetModelFieldType,
});
export type ControlNetModelFieldValue = z.infer<typeof zControlNetModelFieldValue>;
export type ControlNetModelFieldInputInstance = z.infer<typeof zControlNetModelFieldInputInstance>;
export type ControlNetModelFieldInputTemplate = z.infer<typeof zControlNetModelFieldInputTemplate>;
export const isControlNetModelFieldInputInstance = (val: unknown): val is ControlNetModelFieldInputInstance =>
  zControlNetModelFieldInputInstance.safeParse(val).success;
export const isControlNetModelFieldInputTemplate = (val: unknown): val is ControlNetModelFieldInputTemplate =>
  zControlNetModelFieldInputTemplate.safeParse(val).success;
// #endregion

// #region IPAdapterModelField
const zIPAdapterModelFieldType = zFieldTypeBase.extend({
  name: z.literal('IPAdapterModelField'),
});
export const zIPAdapterModelFieldValue = zModelIdentifierField.optional();
const zIPAdapterModelFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zIPAdapterModelFieldValue,
});
const zIPAdapterModelFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zIPAdapterModelFieldType,
  default: zIPAdapterModelFieldValue,
});
const zIPAdapterModelFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zIPAdapterModelFieldType,
});
export type IPAdapterModelFieldValue = z.infer<typeof zIPAdapterModelFieldValue>;
export type IPAdapterModelFieldInputInstance = z.infer<typeof zIPAdapterModelFieldInputInstance>;
export type IPAdapterModelFieldInputTemplate = z.infer<typeof zIPAdapterModelFieldInputTemplate>;
export const isIPAdapterModelFieldInputInstance = (val: unknown): val is IPAdapterModelFieldInputInstance =>
  zIPAdapterModelFieldInputInstance.safeParse(val).success;
export const isIPAdapterModelFieldInputTemplate = (val: unknown): val is IPAdapterModelFieldInputTemplate =>
  zIPAdapterModelFieldInputTemplate.safeParse(val).success;
// #endregion

// #region T2IAdapterField
const zT2IAdapterModelFieldType = zFieldTypeBase.extend({
  name: z.literal('T2IAdapterModelField'),
});
export const zT2IAdapterModelFieldValue = zModelIdentifierField.optional();
const zT2IAdapterModelFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zT2IAdapterModelFieldValue,
});
const zT2IAdapterModelFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zT2IAdapterModelFieldType,
  default: zT2IAdapterModelFieldValue,
});
const zT2IAdapterModelFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zT2IAdapterModelFieldType,
});
export type T2IAdapterModelFieldValue = z.infer<typeof zT2IAdapterModelFieldValue>;
export type T2IAdapterModelFieldInputInstance = z.infer<typeof zT2IAdapterModelFieldInputInstance>;
export type T2IAdapterModelFieldInputTemplate = z.infer<typeof zT2IAdapterModelFieldInputTemplate>;
export const isT2IAdapterModelFieldInputInstance = (val: unknown): val is T2IAdapterModelFieldInputInstance =>
  zT2IAdapterModelFieldInputInstance.safeParse(val).success;
export const isT2IAdapterModelFieldInputTemplate = (val: unknown): val is T2IAdapterModelFieldInputTemplate =>
  zT2IAdapterModelFieldInputTemplate.safeParse(val).success;
// #endregion

// #region SchedulerField
const zSchedulerFieldType = zFieldTypeBase.extend({
  name: z.literal('SchedulerField'),
});
export const zSchedulerFieldValue = zSchedulerField.optional();
const zSchedulerFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zSchedulerFieldValue,
});
const zSchedulerFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zSchedulerFieldType,
  default: zSchedulerFieldValue,
});
const zSchedulerFieldOutputTemplate = zFieldOutputTemplateBase.extend({
  type: zSchedulerFieldType,
});
export type SchedulerFieldValue = z.infer<typeof zSchedulerFieldValue>;
export type SchedulerFieldInputInstance = z.infer<typeof zSchedulerFieldInputInstance>;
export type SchedulerFieldInputTemplate = z.infer<typeof zSchedulerFieldInputTemplate>;
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
const zStatelessFieldType = zFieldTypeBase.extend({
  name: z.string().min(1), // stateless --> we accept the field's name as the type
});
const zStatelessFieldValue = z.undefined().catch(undefined); // stateless --> no value, but making this z.never() introduces a lot of extra TS fanagling
const zStatelessFieldInputInstance = zFieldInputInstanceBase.extend({
  value: zStatelessFieldValue,
});
const zStatelessFieldInputTemplate = zFieldInputTemplateBase.extend({
  type: zStatelessFieldType,
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

// #region StatefulFieldType & FieldType
const zStatefulFieldType = z.union([
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

const zFieldType = z.union([zStatefulFieldType, zStatelessFieldType]);
export type FieldType = z.infer<typeof zFieldType>;
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

const zFieldValue = z.union([zStatefulFieldValue, zStatelessFieldValue]);
export type FieldValue = z.infer<typeof zFieldValue>;
// #endregion

// #region StatefulFieldInputInstance & FieldInputInstance
const zStatefulFieldInputInstance = z.union([
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

export const zFieldInputInstance = z.union([zStatefulFieldInputInstance, zStatelessFieldInputInstance]);
export type FieldInputInstance = z.infer<typeof zFieldInputInstance>;
export const isFieldInputInstance = (val: unknown): val is FieldInputInstance =>
  zFieldInputInstance.safeParse(val).success;
// #endregion

// #region StatefulFieldInputTemplate & FieldInputTemplate
const zStatefulFieldInputTemplate = z.union([
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

export const zFieldInputTemplate = z.union([zStatefulFieldInputTemplate, zStatelessFieldInputTemplate]);
export type FieldInputTemplate = z.infer<typeof zFieldInputTemplate>;
export const isFieldInputTemplate = (val: unknown): val is FieldInputTemplate =>
  zFieldInputTemplate.safeParse(val).success;
// #endregion

// #region StatefulFieldOutputTemplate & FieldOutputTemplate
const zStatefulFieldOutputTemplate = z.union([
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

export const zFieldOutputTemplate = z.union([zStatefulFieldOutputTemplate, zStatelessFieldOutputTemplate]);
export type FieldOutputTemplate = z.infer<typeof zFieldOutputTemplate>;
// #endregion
