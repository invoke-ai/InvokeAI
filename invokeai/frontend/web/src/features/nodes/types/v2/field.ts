import { z } from 'zod';

import {
  zBoardField,
  zCLIPVisionModelField,
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
const zFieldInstanceBase = z.object({
  id: z.string().trim().min(1),
  name: z.string().trim().min(1),
});
const zFieldInputInstanceBase = zFieldInstanceBase.extend({
  fieldKind: z.literal('input'),
  label: z.string().nullish(),
});
const zFieldOutputInstanceBase = zFieldInstanceBase.extend({
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
// #endregion

// #region IntegerField
const zIntegerFieldType = zFieldTypeBase.extend({
  name: z.literal('IntegerField'),
});
const zIntegerFieldValue = z.number().int();
const zIntegerFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zIntegerFieldType,
  value: zIntegerFieldValue,
});
const zIntegerFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zIntegerFieldType,
});
// #endregion

// #region FloatField
const zFloatFieldType = zFieldTypeBase.extend({
  name: z.literal('FloatField'),
});
const zFloatFieldValue = z.number();
const zFloatFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zFloatFieldType,
  value: zFloatFieldValue,
});
const zFloatFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zFloatFieldType,
});
// #endregion

// #region StringField
const zStringFieldType = zFieldTypeBase.extend({
  name: z.literal('StringField'),
});
const zStringFieldValue = z.string();
const zStringFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zStringFieldType,
  value: zStringFieldValue,
});
const zStringFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zStringFieldType,
});
// #endregion

// #region BooleanField
const zBooleanFieldType = zFieldTypeBase.extend({
  name: z.literal('BooleanField'),
});
const zBooleanFieldValue = z.boolean();
const zBooleanFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zBooleanFieldType,
  value: zBooleanFieldValue,
});
const zBooleanFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zBooleanFieldType,
});
// #endregion

// #region EnumField
const zEnumFieldType = zFieldTypeBase.extend({
  name: z.literal('EnumField'),
});
const zEnumFieldValue = z.string();
const zEnumFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zEnumFieldType,
  value: zEnumFieldValue,
});
const zEnumFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zEnumFieldType,
});
// #endregion

// #region ImageField
const zImageFieldType = zFieldTypeBase.extend({
  name: z.literal('ImageField'),
});
const zImageFieldValue = zImageField.optional();
const zImageFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zImageFieldType,
  value: zImageFieldValue,
});
const zImageFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zImageFieldType,
});
// #endregion

// #region BoardField
const zBoardFieldType = zFieldTypeBase.extend({
  name: z.literal('BoardField'),
});
const zBoardFieldValue = zBoardField.optional();
const zBoardFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zBoardFieldType,
  value: zBoardFieldValue,
});
const zBoardFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zBoardFieldType,
});
// #endregion

// #region ColorField
const zColorFieldType = zFieldTypeBase.extend({
  name: z.literal('ColorField'),
});
const zColorFieldValue = zColorField.optional();
const zColorFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zColorFieldType,
  value: zColorFieldValue,
});
const zColorFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zColorFieldType,
});
// #endregion

// #region MainModelField
const zMainModelFieldType = zFieldTypeBase.extend({
  name: z.literal('MainModelField'),
});
const zMainModelFieldValue = zMainModelField.optional();
const zMainModelFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zMainModelFieldType,
  value: zMainModelFieldValue,
});
const zMainModelFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zMainModelFieldType,
});
// #endregion

// #region SDXLMainModelField
const zSDXLMainModelFieldType = zFieldTypeBase.extend({
  name: z.literal('SDXLMainModelField'),
});
const zSDXLMainModelFieldValue = zMainModelFieldValue; // TODO: Narrow to SDXL models only.
const zSDXLMainModelFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zSDXLMainModelFieldType,
  value: zSDXLMainModelFieldValue,
});
const zSDXLMainModelFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zSDXLMainModelFieldType,
});
// #endregion

// #region SDXLRefinerModelField
const zSDXLRefinerModelFieldType = zFieldTypeBase.extend({
  name: z.literal('SDXLRefinerModelField'),
});
const zSDXLRefinerModelFieldValue = zMainModelFieldValue; // TODO: Narrow to SDXL Refiner models only.
const zSDXLRefinerModelFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zSDXLRefinerModelFieldType,
  value: zSDXLRefinerModelFieldValue,
});
const zSDXLRefinerModelFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zSDXLRefinerModelFieldType,
});
// #endregion

// #region VAEModelField
const zVAEModelFieldType = zFieldTypeBase.extend({
  name: z.literal('VAEModelField'),
});
const zVAEModelFieldValue = zVAEModelField.optional();
const zVAEModelFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zVAEModelFieldType,
  value: zVAEModelFieldValue,
});
const zVAEModelFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zVAEModelFieldType,
});
// #endregion

// #region CLIPVisionModelField
const zCLIPVisionModelFieldType = zFieldTypeBase.extend({
  name: z.literal('CLIPVisionModelField'),
});
const zCLIPVisionModelFieldValue = zCLIPVisionModelField.optional();
const zCLIPVisionModelFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zCLIPVisionModelFieldType,
  value: zCLIPVisionModelFieldValue,
});
const zCLIPVisionModelFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zCLIPVisionModelFieldType,
});
// #endregion

// #region LoRAModelField
const zLoRAModelFieldType = zFieldTypeBase.extend({
  name: z.literal('LoRAModelField'),
});
const zLoRAModelFieldValue = zLoRAModelField.optional();
const zLoRAModelFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zLoRAModelFieldType,
  value: zLoRAModelFieldValue,
});
const zLoRAModelFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zLoRAModelFieldType,
});
// #endregion

// #region ControlNetModelField
const zControlNetModelFieldType = zFieldTypeBase.extend({
  name: z.literal('ControlNetModelField'),
});
const zControlNetModelFieldValue = zControlNetModelField.optional();
const zControlNetModelFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zControlNetModelFieldType,
  value: zControlNetModelFieldValue,
});
const zControlNetModelFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zControlNetModelFieldType,
});
// #endregion

// #region IPAdapterModelField
const zIPAdapterModelFieldType = zFieldTypeBase.extend({
  name: z.literal('IPAdapterModelField'),
});
const zIPAdapterModelFieldValue = zIPAdapterModelField.optional();
const zIPAdapterModelFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zIPAdapterModelFieldType,
  value: zIPAdapterModelFieldValue,
});
const zIPAdapterModelFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zIPAdapterModelFieldType,
});
// #endregion

// #region T2IAdapterField
const zT2IAdapterModelFieldType = zFieldTypeBase.extend({
  name: z.literal('T2IAdapterModelField'),
});
const zT2IAdapterModelFieldValue = zT2IAdapterModelField.optional();
const zT2IAdapterModelFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zT2IAdapterModelFieldType,
  value: zT2IAdapterModelFieldValue,
});
const zT2IAdapterModelFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zT2IAdapterModelFieldType,
});
// #endregion

// #region SchedulerField
const zSchedulerFieldType = zFieldTypeBase.extend({
  name: z.literal('SchedulerField'),
});
const zSchedulerFieldValue = zSchedulerField.optional();
const zSchedulerFieldInputInstance = zFieldInputInstanceBase.extend({
  type: zSchedulerFieldType,
  value: zSchedulerFieldValue,
});
const zSchedulerFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zSchedulerFieldType,
});
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
  type: zStatelessFieldType,
  value: zStatelessFieldValue,
});
const zStatelessFieldOutputInstance = zFieldOutputInstanceBase.extend({
  type: zStatelessFieldType,
});

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
  zCLIPVisionModelFieldInputInstance,
  zLoRAModelFieldInputInstance,
  zControlNetModelFieldInputInstance,
  zIPAdapterModelFieldInputInstance,
  zT2IAdapterModelFieldInputInstance,
  zColorFieldInputInstance,
  zSchedulerFieldInputInstance,
]);

export const zFieldInputInstance = z.union([zStatefulFieldInputInstance, zStatelessFieldInputInstance]);
// #endregion

// #region StatefulFieldOutputInstance & FieldOutputInstance
const zStatefulFieldOutputInstance = z.union([
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
  zCLIPVisionModelFieldOutputInstance,
  zLoRAModelFieldOutputInstance,
  zControlNetModelFieldOutputInstance,
  zIPAdapterModelFieldOutputInstance,
  zT2IAdapterModelFieldOutputInstance,
  zColorFieldOutputInstance,
  zSchedulerFieldOutputInstance,
]);

export const zFieldOutputInstance = z.union([zStatefulFieldOutputInstance, zStatelessFieldOutputInstance]);
// #endregion
