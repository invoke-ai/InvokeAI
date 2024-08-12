import { z } from 'zod';

// WorkflowV1 Schema

const zScheduler = z.enum([
  'euler',
  'deis',
  'ddim',
  'ddpm',
  'dpmpp_2s',
  'dpmpp_2m',
  'dpmpp_2m_sde',
  'dpmpp_sde',
  'heun',
  'kdpm_2',
  'lms',
  'pndm',
  'unipc',
  'euler_k',
  'dpmpp_2s_k',
  'dpmpp_2m_k',
  'dpmpp_2m_sde_k',
  'dpmpp_sde_k',
  'heun_k',
  'lms_k',
  'euler_a',
  'kdpm_2_a',
  'lcm',
]);
const zBaseModel = z.enum(['any', 'sd-1', 'sd-2', 'sdxl', 'sdxl-refiner']);
const zMainModel = z.object({
  model_name: z.string().min(1),
  base_model: zBaseModel,
  model_type: z.literal('main'),
});
const zOnnxModel = z.object({
  model_name: z.string().min(1),
  base_model: zBaseModel,
  model_type: z.literal('onnx'),
});

const zMainOrOnnxModel = z.union([zMainModel, zOnnxModel]);

// TODO: Get this from the OpenAPI schema? may be tricky...
const zFieldTypeV1 = z.enum([
  'Any',
  'BoardField',
  'boolean',
  'BooleanCollection',
  'BooleanPolymorphic',
  'ClipField',
  'Collection',
  'CollectionItem',
  'ColorCollection',
  'ColorField',
  'ColorPolymorphic',
  'ConditioningCollection',
  'ConditioningField',
  'ConditioningPolymorphic',
  'ControlCollection',
  'ControlField',
  'ControlNetModelField',
  'ControlPolymorphic',
  'DenoiseMaskField',
  'enum',
  'float',
  'FloatCollection',
  'FloatPolymorphic',
  'ImageCollection',
  'ImageField',
  'ImagePolymorphic',
  'integer',
  'IntegerCollection',
  'IntegerPolymorphic',
  'IPAdapterCollection',
  'IPAdapterField',
  'IPAdapterModelField',
  'IPAdapterPolymorphic',
  'LatentsCollection',
  'LatentsField',
  'LatentsPolymorphic',
  'LoRAModelField',
  'MainModelField',
  'MetadataField',
  'MetadataCollection',
  'MetadataItemField',
  'MetadataItemCollection',
  'MetadataItemPolymorphic',
  'ONNXModelField',
  'Scheduler',
  'SDXLMainModelField',
  'SDXLRefinerModelField',
  'string',
  'StringCollection',
  'StringPolymorphic',
  'T2IAdapterCollection',
  'T2IAdapterField',
  'T2IAdapterModelField',
  'T2IAdapterPolymorphic',
  'UNetField',
  'VaeField',
  'VaeModelField',
]);
export type FieldTypeV1 = z.infer<typeof zFieldTypeV1>;

const zFieldValueBase = z.object({
  id: z.string().trim().min(1),
  name: z.string().trim().min(1),
  type: zFieldTypeV1,
});

/**
 * An output field is persisted across as part of the user's local state.
 *
 * An output field has two properties:
 * - `id` a unique identifier
 * - `name` the name of the field, which comes from the python dataclass
 */

const zOutputFieldValue = zFieldValueBase.extend({
  fieldKind: z.literal('output'),
});

const zInputFieldValueBase = zFieldValueBase.extend({
  fieldKind: z.literal('input'),
  label: z.string(),
});

const zModelIdentifier = z.object({
  model_name: z.string().trim().min(1),
  base_model: zBaseModel,
});

const zImageField = z.object({
  image_name: z.string().trim().min(1),
});

const zBoardField = z.object({
  board_id: z.string().trim().min(1),
});

const zLatentsField = z.object({
  latents_name: z.string().trim().min(1),
  seed: z.number().int().optional(),
});

const zConditioningField = z.object({
  conditioning_name: z.string().trim().min(1),
});

const zDenoiseMaskField = z.object({
  mask_name: z.string().trim().min(1),
  masked_latents_name: z.string().trim().min(1).optional(),
});

const zIntegerInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('integer'),
  value: z.number().int().optional(),
});

const zIntegerCollectionInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('IntegerCollection'),
  value: z.array(z.number().int()).optional(),
});

const zIntegerPolymorphicInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('IntegerPolymorphic'),
  value: z.number().int().optional(),
});

const zFloatInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('float'),
  value: z.number().optional(),
});

const zFloatCollectionInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('FloatCollection'),
  value: z.array(z.number()).optional(),
});

const zFloatPolymorphicInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('FloatPolymorphic'),
  value: z.number().optional(),
});

const zStringInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('string'),
  value: z.string().optional(),
});

const zStringCollectionInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('StringCollection'),
  value: z.array(z.string()).optional(),
});

const zStringPolymorphicInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('StringPolymorphic'),
  value: z.string().optional(),
});

const zBooleanInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('boolean'),
  value: z.boolean().optional(),
});

const zBooleanCollectionInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('BooleanCollection'),
  value: z.array(z.boolean()).optional(),
});

const zBooleanPolymorphicInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('BooleanPolymorphic'),
  value: z.boolean().optional(),
});

const zEnumInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('enum'),
  value: z.string().optional(),
});

const zLatentsInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('LatentsField'),
  value: zLatentsField.optional(),
});

const zLatentsCollectionInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('LatentsCollection'),
  value: z.array(zLatentsField).optional(),
});

const zLatentsPolymorphicInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('LatentsPolymorphic'),
  value: z.union([zLatentsField, z.array(zLatentsField)]).optional(),
});

const zDenoiseMaskInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('DenoiseMaskField'),
  value: zDenoiseMaskField.optional(),
});

const zConditioningInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('ConditioningField'),
  value: zConditioningField.optional(),
});

const zConditioningCollectionInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('ConditioningCollection'),
  value: z.array(zConditioningField).optional(),
});

const zConditioningPolymorphicInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('ConditioningPolymorphic'),
  value: z.union([zConditioningField, z.array(zConditioningField)]).optional(),
});

const zControlNetModel = zModelIdentifier;

const zControlField = z.object({
  image: zImageField,
  control_model: zControlNetModel,
  control_weight: z.union([z.number(), z.array(z.number())]).optional(),
  begin_step_percent: z.number().optional(),
  end_step_percent: z.number().optional(),
  control_mode: z.enum(['balanced', 'more_prompt', 'more_control', 'unbalanced']).optional(),
  resize_mode: z.enum(['just_resize', 'crop_resize', 'fill_resize', 'just_resize_simple']).optional(),
});

const zControlInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('ControlField'),
  value: zControlField.optional(),
});

const zControlPolymorphicInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('ControlPolymorphic'),
  value: z.union([zControlField, z.array(zControlField)]).optional(),
});

const zControlCollectionInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('ControlCollection'),
  value: z.array(zControlField).optional(),
});

const zIPAdapterModel = zModelIdentifier;

const zIPAdapterField = z.object({
  image: zImageField,
  ip_adapter_model: zIPAdapterModel,
  weight: z.number(),
  begin_step_percent: z.number().optional(),
  end_step_percent: z.number().optional(),
});

const zIPAdapterInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('IPAdapterField'),
  value: zIPAdapterField.optional(),
});

const zIPAdapterPolymorphicInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('IPAdapterPolymorphic'),
  value: z.union([zIPAdapterField, z.array(zIPAdapterField)]).optional(),
});

const zIPAdapterCollectionInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('IPAdapterCollection'),
  value: z.array(zIPAdapterField).optional(),
});

const zT2IAdapterModel = zModelIdentifier;

const zT2IAdapterField = z.object({
  image: zImageField,
  t2i_adapter_model: zT2IAdapterModel,
  weight: z.union([z.number(), z.array(z.number())]).optional(),
  begin_step_percent: z.number().optional(),
  end_step_percent: z.number().optional(),
  resize_mode: z.enum(['just_resize', 'crop_resize', 'fill_resize', 'just_resize_simple']).optional(),
});

const zT2IAdapterInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('T2IAdapterField'),
  value: zT2IAdapterField.optional(),
});

const zT2IAdapterPolymorphicInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('T2IAdapterPolymorphic'),
  value: z.union([zT2IAdapterField, z.array(zT2IAdapterField)]).optional(),
});

const zT2IAdapterCollectionInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('T2IAdapterCollection'),
  value: z.array(zT2IAdapterField).optional(),
});

const zModelType = z.enum(['onnx', 'main', 'vae', 'lora', 'controlnet', 'embedding']);

const zSubModelType = z.enum([
  'unet',
  'text_encoder',
  'text_encoder_2',
  'tokenizer',
  'tokenizer_2',
  'vae',
  'vae_decoder',
  'vae_encoder',
  'scheduler',
  'safety_checker',
]);

const zModelInfo = zModelIdentifier.extend({
  model_type: zModelType,
  submodel: zSubModelType.optional(),
});

const zLoraInfo = zModelInfo.extend({
  weight: z.number().optional(),
});

const zUNetField = z.object({
  unet: zModelInfo,
  scheduler: zModelInfo,
  loras: z.array(zLoraInfo),
});

const zUNetInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('UNetField'),
  value: zUNetField.optional(),
});

const zClipField = z.object({
  tokenizer: zModelInfo,
  text_encoder: zModelInfo,
  skipped_layers: z.number(),
  loras: z.array(zLoraInfo),
});

const zClipInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('ClipField'),
  value: zClipField.optional(),
});

const zVaeField = z.object({
  vae: zModelInfo,
});

const zVaeInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('VaeField'),
  value: zVaeField.optional(),
});

const zImageInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('ImageField'),
  value: zImageField.optional(),
});

const zBoardInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('BoardField'),
  value: zBoardField.optional(),
});

const zImagePolymorphicInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('ImagePolymorphic'),
  value: zImageField.optional(),
});

const zImageCollectionInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('ImageCollection'),
  value: z.array(zImageField).optional(),
});

const zMainModelInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('MainModelField'),
  value: zMainOrOnnxModel.optional(),
});

const zSDXLMainModelInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('SDXLMainModelField'),
  value: zMainOrOnnxModel.optional(),
});

const zSDXLRefinerModelInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('SDXLRefinerModelField'),
  value: zMainOrOnnxModel.optional(), // TODO: should narrow this down to a refiner model
});

const zVaeModelField = zModelIdentifier;

const zVaeModelInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('VaeModelField'),
  value: zVaeModelField.optional(),
});

const zLoRAModelField = zModelIdentifier;

const zLoRAModelInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('LoRAModelField'),
  value: zLoRAModelField.optional(),
});

const zControlNetModelField = zModelIdentifier;

const zControlNetModelInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('ControlNetModelField'),
  value: zControlNetModelField.optional(),
});

const zIPAdapterModelField = zModelIdentifier;

const zIPAdapterModelInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('IPAdapterModelField'),
  value: zIPAdapterModelField.optional(),
});

const zT2IAdapterModelField = zModelIdentifier;

const zT2IAdapterModelInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('T2IAdapterModelField'),
  value: zT2IAdapterModelField.optional(),
});

const zCollectionInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('Collection'),
  value: z.array(z.any()).optional(), // TODO: should this field ever have a value?
});

const zCollectionItemInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('CollectionItem'),
  value: z.any().optional(), // TODO: should this field ever have a value?
});

const zMetadataItemField = z.object({
  label: z.string(),
  value: z.any(),
});

const zMetadataItemInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('MetadataItemField'),
  value: zMetadataItemField.optional(),
});

const zMetadataItemCollectionInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('MetadataItemCollection'),
  value: z.array(zMetadataItemField).optional(),
});

const zMetadataItemPolymorphicInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('MetadataItemPolymorphic'),
  value: z.union([zMetadataItemField, z.array(zMetadataItemField)]).optional(),
});

const zMetadataField = z.record(z.any());

const zMetadataInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('MetadataField'),
  value: zMetadataField.optional(),
});

const zMetadataCollectionInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('MetadataCollection'),
  value: z.array(zMetadataField).optional(),
});

const zColorField = z.object({
  r: z.number().int().min(0).max(255),
  g: z.number().int().min(0).max(255),
  b: z.number().int().min(0).max(255),
  a: z.number().int().min(0).max(255),
});

const zColorInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('ColorField'),
  value: zColorField.optional(),
});

const zColorCollectionInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('ColorCollection'),
  value: z.array(zColorField).optional(),
});

const zColorPolymorphicInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('ColorPolymorphic'),
  value: z.union([zColorField, z.array(zColorField)]).optional(),
});

const zSchedulerInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('Scheduler'),
  value: zScheduler.optional(),
});

const zAnyInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('Any'),
  value: z.any().optional(),
});

const zInputFieldValue = z.discriminatedUnion('type', [
  zAnyInputFieldValue,
  zBoardInputFieldValue,
  zBooleanCollectionInputFieldValue,
  zBooleanInputFieldValue,
  zBooleanPolymorphicInputFieldValue,
  zClipInputFieldValue,
  zCollectionInputFieldValue,
  zCollectionItemInputFieldValue,
  zColorInputFieldValue,
  zColorCollectionInputFieldValue,
  zColorPolymorphicInputFieldValue,
  zConditioningInputFieldValue,
  zConditioningCollectionInputFieldValue,
  zConditioningPolymorphicInputFieldValue,
  zControlInputFieldValue,
  zControlNetModelInputFieldValue,
  zControlCollectionInputFieldValue,
  zControlPolymorphicInputFieldValue,
  zDenoiseMaskInputFieldValue,
  zEnumInputFieldValue,
  zFloatCollectionInputFieldValue,
  zFloatInputFieldValue,
  zFloatPolymorphicInputFieldValue,
  zImageCollectionInputFieldValue,
  zImagePolymorphicInputFieldValue,
  zImageInputFieldValue,
  zIntegerCollectionInputFieldValue,
  zIntegerPolymorphicInputFieldValue,
  zIntegerInputFieldValue,
  zIPAdapterInputFieldValue,
  zIPAdapterModelInputFieldValue,
  zIPAdapterCollectionInputFieldValue,
  zIPAdapterPolymorphicInputFieldValue,
  zLatentsInputFieldValue,
  zLatentsCollectionInputFieldValue,
  zLatentsPolymorphicInputFieldValue,
  zLoRAModelInputFieldValue,
  zMainModelInputFieldValue,
  zSchedulerInputFieldValue,
  zSDXLMainModelInputFieldValue,
  zSDXLRefinerModelInputFieldValue,
  zStringCollectionInputFieldValue,
  zStringPolymorphicInputFieldValue,
  zStringInputFieldValue,
  zT2IAdapterInputFieldValue,
  zT2IAdapterModelInputFieldValue,
  zT2IAdapterCollectionInputFieldValue,
  zT2IAdapterPolymorphicInputFieldValue,
  zUNetInputFieldValue,
  zVaeInputFieldValue,
  zVaeModelInputFieldValue,
  zMetadataItemInputFieldValue,
  zMetadataItemCollectionInputFieldValue,
  zMetadataItemPolymorphicInputFieldValue,
  zMetadataInputFieldValue,
  zMetadataCollectionInputFieldValue,
]);

const zSemVer = z.string().refine((val) => {
  const [major, minor, patch] = val.split('.');
  return (
    major !== undefined &&
    Number.isInteger(Number(major)) &&
    minor !== undefined &&
    Number.isInteger(Number(minor)) &&
    patch !== undefined &&
    Number.isInteger(Number(patch))
  );
});

const zInvocationNodeData = z.object({
  id: z.string().trim().min(1),
  // no easy way to build this dynamically, and we don't want to anyways, because this will be used
  // to validate incoming workflows, and we want to allow community nodes.
  type: z.string().trim().min(1),
  inputs: z.record(zInputFieldValue),
  outputs: z.record(zOutputFieldValue),
  label: z.string(),
  isOpen: z.boolean(),
  notes: z.string(),
  embedWorkflow: z.boolean(),
  isIntermediate: z.boolean(),
  useCache: z.boolean().default(true),
  version: zSemVer.optional(),
});

const zNotesNodeData = z.object({
  id: z.string().trim().min(1),
  type: z.literal('notes'),
  label: z.string(),
  isOpen: z.boolean(),
  notes: z.string(),
});

const zPosition = z
  .object({
    x: z.number(),
    y: z.number(),
  })
  .default({ x: 0, y: 0 });

const zDimension = z.number().gt(0).nullish();

const zWorkflowInvocationNode = z.object({
  id: z.string().trim().min(1),
  type: z.literal('invocation'),
  data: zInvocationNodeData,
  width: zDimension,
  height: zDimension,
  position: zPosition,
});

const zWorkflowNotesNode = z.object({
  id: z.string().trim().min(1),
  type: z.literal('notes'),
  data: zNotesNodeData,
  width: zDimension,
  height: zDimension,
  position: zPosition,
});

const zWorkflowNode = z.discriminatedUnion('type', [zWorkflowInvocationNode, zWorkflowNotesNode]);

const zDefaultWorkflowEdge = z.object({
  source: z.string().trim().min(1),
  sourceHandle: z.string().trim().min(1),
  target: z.string().trim().min(1),
  targetHandle: z.string().trim().min(1),
  id: z.string().trim().min(1),
  type: z.literal('default'),
});
const zCollapsedWorkflowEdge = z.object({
  source: z.string().trim().min(1),
  target: z.string().trim().min(1),
  id: z.string().trim().min(1),
  type: z.literal('collapsed'),
});

const zWorkflowEdge = z.union([zDefaultWorkflowEdge, zCollapsedWorkflowEdge]);

const zFieldIdentifier = z.object({
  nodeId: z.string().trim().min(1),
  fieldName: z.string().trim().min(1),
});

export const zWorkflowV1 = z.object({
  name: z.string().default(''),
  author: z.string().default(''),
  description: z.string().default(''),
  version: z.string().default(''),
  contact: z.string().default(''),
  tags: z.string().default(''),
  notes: z.string().default(''),
  nodes: z.array(zWorkflowNode).default([]),
  edges: z.array(zWorkflowEdge).default([]),
  exposedFields: z.array(zFieldIdentifier).default([]),
  meta: z.object({
    version: z.literal('1.0.0'),
  }),
});
export type WorkflowV1 = z.infer<typeof zWorkflowV1>;
