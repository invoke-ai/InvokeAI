import { store } from 'app/store/store';
import {
  SchedulerParam,
  zBaseModel,
  zMainOrOnnxModel,
  zSDXLRefinerModel,
  zScheduler,
} from 'features/parameters/types/parameterSchemas';
import { keyBy } from 'lodash-es';
import { OpenAPIV3 } from 'openapi-types';
import { RgbaColor } from 'react-colorful';
import { Node } from 'reactflow';
import { JsonObject } from 'type-fest';
import { Graph, ImageDTO, _InputField, _OutputField } from 'services/api/types';
import {
  AnyInvocationType,
  AnyResult,
  ProgressImage,
} from 'services/events/types';
import { O } from 'ts-toolbelt';
import { z } from 'zod';

export type NonNullableGraph = O.Required<Graph, 'nodes' | 'edges'>;

export type InvocationTemplate = {
  /**
   * Unique type of the invocation
   */
  type: AnyInvocationType;
  /**
   * Display name of the invocation
   */
  title: string;
  /**
   * Description of the invocation
   */
  description: string;
  /**
   * Invocation tags
   */
  tags: string[];
  /**
   * Array of invocation inputs
   */
  inputs: Record<string, InputFieldTemplate>;
  /**
   * Array of the invocation outputs
   */
  outputs: Record<string, OutputFieldTemplate>;
  /**
   * The type of this node's output
   */
  outputType: string; // TODO: generate a union of output types
};

export type FieldUIConfig = {
  title: string;
  description: string;
  color: string;
};

// TODO: Get this from the OpenAPI schema? may be tricky...
export const zFieldType = z.enum([
  // region Primitives
  'integer',
  'float',
  'boolean',
  'string',
  'array',
  'ImageField',
  'DenoiseMaskField',
  'LatentsField',
  'ConditioningField',
  'ControlField',
  'ColorField',
  'ImageCollection',
  'ConditioningCollection',
  'ColorCollection',
  'LatentsCollection',
  'IntegerCollection',
  'FloatCollection',
  'StringCollection',
  'BooleanCollection',
  // endregion

  // region Models
  'MainModelField',
  'SDXLMainModelField',
  'SDXLRefinerModelField',
  'ONNXModelField',
  'VaeModelField',
  'LoRAModelField',
  'ControlNetModelField',
  'UNetField',
  'VaeField',
  'ClipField',
  // endregion

  // region Iterate/Collect
  'Collection',
  'CollectionItem',
  // endregion

  // region Misc
  'enum',
  'Scheduler',
  // endregion
]);

export type FieldType = z.infer<typeof zFieldType>;

export const zReservedFieldType = z.enum([
  'WorkflowField',
  'IsIntermediate',
  'MetadataField',
]);

export type ReservedFieldType = z.infer<typeof zReservedFieldType>;

export const isFieldType = (value: unknown): value is FieldType =>
  zFieldType.safeParse(value).success ||
  zReservedFieldType.safeParse(value).success;

/**
 * An input field template is generated on each page load from the OpenAPI schema.
 *
 * The template provides the field type and other field metadata (e.g. title, description,
 * maximum length, pattern to match, etc).
 */
export type InputFieldTemplate =
  | IntegerInputFieldTemplate
  | FloatInputFieldTemplate
  | StringInputFieldTemplate
  | BooleanInputFieldTemplate
  | ImageInputFieldTemplate
  | DenoiseMaskInputFieldTemplate
  | LatentsInputFieldTemplate
  | ConditioningInputFieldTemplate
  | UNetInputFieldTemplate
  | ClipInputFieldTemplate
  | VaeInputFieldTemplate
  | ControlInputFieldTemplate
  | EnumInputFieldTemplate
  | MainModelInputFieldTemplate
  | SDXLMainModelInputFieldTemplate
  | SDXLRefinerModelInputFieldTemplate
  | VaeModelInputFieldTemplate
  | LoRAModelInputFieldTemplate
  | ControlNetModelInputFieldTemplate
  | CollectionInputFieldTemplate
  | CollectionItemInputFieldTemplate
  | ColorInputFieldTemplate
  | ImageCollectionInputFieldTemplate
  | SchedulerInputFieldTemplate;

/**
 * Indicates the kind of input(s) this field may have.
 */
export const zInputKind = z.enum(['connection', 'direct', 'any']);
export type InputKind = z.infer<typeof zInputKind>;

export const zFieldValueBase = z.object({
  id: z.string().trim().min(1),
  name: z.string().trim().min(1),
  type: zFieldType,
});
export type FieldValueBase = z.infer<typeof zFieldValueBase>;

/**
 * An output field is persisted across as part of the user's local state.
 *
 * An output field has two properties:
 * - `id` a unique identifier
 * - `name` the name of the field, which comes from the python dataclass
 */

export const zOutputFieldValue = zFieldValueBase.extend({
  fieldKind: z.literal('output'),
});
export type OutputFieldValue = z.infer<typeof zOutputFieldValue>;

/**
 * An output field template is generated on each page load from the OpenAPI schema.
 *
 * The template provides the output field's name, type, title, and description.
 */
export type OutputFieldTemplate = {
  fieldKind: 'output';
  name: string;
  type: FieldType;
  title: string;
  description: string;
} & _OutputField;

export const zInputFieldValueBase = zFieldValueBase.extend({
  fieldKind: z.literal('input'),
  label: z.string(),
});
export type InputFieldValueBase = z.infer<typeof zInputFieldValueBase>;

export const zModelIdentifier = z.object({
  model_name: z.string().trim().min(1),
  base_model: zBaseModel,
});

export const zImageField = z.object({
  image_name: z.string().trim().min(1),
});
export type ImageField = z.infer<typeof zImageField>;

export const zLatentsField = z.object({
  latents_name: z.string().trim().min(1),
  seed: z.number().int().optional(),
});
export type LatentsField = z.infer<typeof zLatentsField>;

export const zConditioningField = z.object({
  conditioning_name: z.string().trim().min(1),
});
export type ConditioningField = z.infer<typeof zConditioningField>;

export const zDenoiseMaskField = z.object({
  mask_name: z.string().trim().min(1),
  masked_latents_name: z.string().trim().min(1).optional(),
});
export type DenoiseMaskFieldValue = z.infer<typeof zDenoiseMaskField>;

export const zIntegerInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('integer'),
  value: z.number().int().optional(),
});
export type IntegerInputFieldValue = z.infer<typeof zIntegerInputFieldValue>;

export const zFloatInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('float'),
  value: z.number().optional(),
});
export type FloatInputFieldValue = z.infer<typeof zFloatInputFieldValue>;

export const zStringInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('string'),
  value: z.string().optional(),
});
export type StringInputFieldValue = z.infer<typeof zStringInputFieldValue>;

export const zBooleanInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('boolean'),
  value: z.boolean().optional(),
});
export type BooleanInputFieldValue = z.infer<typeof zBooleanInputFieldValue>;

export const zEnumInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('enum'),
  value: z.union([z.string(), z.number()]).optional(),
});
export type EnumInputFieldValue = z.infer<typeof zEnumInputFieldValue>;

export const zLatentsInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('LatentsField'),
  value: zLatentsField.optional(),
});
export type LatentsInputFieldValue = z.infer<typeof zLatentsInputFieldValue>;

export const zDenoiseMaskInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('DenoiseMaskField'),
  value: zDenoiseMaskField.optional(),
});
export type DenoiseMaskInputFieldValue = z.infer<
  typeof zDenoiseMaskInputFieldValue
>;

export const zConditioningInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('ConditioningField'),
  value: zConditioningField.optional(),
});
export type ConditioningInputFieldValue = z.infer<
  typeof zConditioningInputFieldValue
>;

export const zControlNetModel = zModelIdentifier;
export type ControlNetModel = z.infer<typeof zControlNetModel>;

export const zControlField = zInputFieldValueBase.extend({
  image: zImageField,
  control_model: zControlNetModel,
  control_weight: z.union([z.number(), z.array(z.number())]).optional(),
  begin_step_percent: z.number().optional(),
  end_step_percent: z.number().optional(),
  control_mode: z
    .enum(['balanced', 'more_prompt', 'more_control', 'unbalanced'])
    .optional(),
  resize_mode: z
    .enum(['just_resize', 'crop_resize', 'fill_resize', 'just_resize_simple'])
    .optional(),
});
export type ControlField = z.infer<typeof zControlField>;

export const zControlInputFieldTemplate = zInputFieldValueBase.extend({
  type: z.literal('ControlField'),
  value: zControlField.optional(),
});
export type ControlInputFieldValue = z.infer<typeof zControlInputFieldTemplate>;

export const zModelType = z.enum([
  'onnx',
  'main',
  'vae',
  'lora',
  'controlnet',
  'embedding',
]);
export type ModelType = z.infer<typeof zModelType>;

export const zSubModelType = z.enum([
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
export type SubModelType = z.infer<typeof zSubModelType>;

export const zModelInfo = zModelIdentifier.extend({
  model_type: zModelType,
  submodel: zSubModelType.optional(),
});
export type ModelInfo = z.infer<typeof zModelInfo>;

export const zLoraInfo = zModelInfo.extend({
  weight: z.number().optional(),
});
export type LoraInfo = z.infer<typeof zLoraInfo>;

export const zUNetField = z.object({
  unet: zModelInfo,
  scheduler: zModelInfo,
  loras: z.array(zLoraInfo),
});
export type UNetField = z.infer<typeof zUNetField>;

export const zUNetInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('UNetField'),
  value: zUNetField.optional(),
});
export type UNetInputFieldValue = z.infer<typeof zUNetInputFieldValue>;

export const zClipField = z.object({
  tokenizer: zModelInfo,
  text_encoder: zModelInfo,
  skipped_layers: z.number(),
  loras: z.array(zLoraInfo),
});
export type ClipField = z.infer<typeof zClipField>;

export const zClipInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('ClipField'),
  value: zClipField.optional(),
});
export type ClipInputFieldValue = z.infer<typeof zClipInputFieldValue>;

export const zVaeField = z.object({
  vae: zModelInfo,
});
export type VaeField = z.infer<typeof zVaeField>;

export const zVaeInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('VaeField'),
  value: zVaeField.optional(),
});
export type VaeInputFieldValue = z.infer<typeof zVaeInputFieldValue>;

export const zImageInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('ImageField'),
  value: zImageField.optional(),
});
export type ImageInputFieldValue = z.infer<typeof zImageInputFieldValue>;

export const zImageCollectionInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('ImageCollection'),
  value: z.array(zImageField).optional(),
});
export type ImageCollectionInputFieldValue = z.infer<
  typeof zImageCollectionInputFieldValue
>;

export const zMainModelInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('MainModelField'),
  value: zMainOrOnnxModel.optional(),
});
export type MainModelInputFieldValue = z.infer<
  typeof zMainModelInputFieldValue
>;

export const zSDXLMainModelInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('SDXLMainModelField'),
  value: zMainOrOnnxModel.optional(),
});
export type SDXLMainModelInputFieldValue = z.infer<
  typeof zSDXLMainModelInputFieldValue
>;

export const zSDXLRefinerModelInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('SDXLRefinerModelField'),
  value: zMainOrOnnxModel.optional(), // TODO: should narrow this down to a refiner model
});
export type SDXLRefinerModelInputFieldValue = z.infer<
  typeof zSDXLRefinerModelInputFieldValue
>;

export const zVaeModelField = zModelIdentifier;

export const zVaeModelInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('VaeModelField'),
  value: zVaeModelField.optional(),
});
export type VaeModelInputFieldValue = z.infer<typeof zVaeModelInputFieldValue>;

export const zLoRAModelField = zModelIdentifier;
export type LoRAModelField = z.infer<typeof zLoRAModelField>;

export const zLoRAModelInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('LoRAModelField'),
  value: zLoRAModelField.optional(),
});
export type LoRAModelInputFieldValue = z.infer<
  typeof zLoRAModelInputFieldValue
>;

export const zControlNetModelField = zModelIdentifier;
export type ControlNetModelField = z.infer<typeof zControlNetModelField>;

export const zControlNetModelInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('ControlNetModelField'),
  value: zControlNetModelField.optional(),
});
export type ControlNetModelInputFieldValue = z.infer<
  typeof zControlNetModelInputFieldValue
>;

export const zCollectionInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('Collection'),
  value: z.array(z.any()).optional(), // TODO: should this field ever have a value?
});
export type CollectionInputFieldValue = z.infer<
  typeof zCollectionInputFieldValue
>;

export const zCollectionItemInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('CollectionItem'),
  value: z.any().optional(), // TODO: should this field ever have a value?
});
export type CollectionItemInputFieldValue = z.infer<
  typeof zCollectionItemInputFieldValue
>;

export const zColorField = z.object({
  r: z.number().int().min(0).max(255),
  g: z.number().int().min(0).max(255),
  b: z.number().int().min(0).max(255),
  a: z.number().int().min(0).max(255),
});
export type ColorField = z.infer<typeof zColorField>;

export const zColorInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('ColorField'),
  value: zColorField.optional(),
});
export type ColorInputFieldValue = z.infer<typeof zColorInputFieldValue>;

export const zSchedulerInputFieldValue = zInputFieldValueBase.extend({
  type: z.literal('Scheduler'),
  value: zScheduler.optional(),
});
export type SchedulerInputFieldValue = z.infer<
  typeof zSchedulerInputFieldValue
>;

export const zInputFieldValue = z.discriminatedUnion('type', [
  zIntegerInputFieldValue,
  zFloatInputFieldValue,
  zStringInputFieldValue,
  zBooleanInputFieldValue,
  zImageInputFieldValue,
  zLatentsInputFieldValue,
  zDenoiseMaskInputFieldValue,
  zConditioningInputFieldValue,
  zUNetInputFieldValue,
  zClipInputFieldValue,
  zVaeInputFieldValue,
  zControlInputFieldTemplate,
  zEnumInputFieldValue,
  zMainModelInputFieldValue,
  zSDXLMainModelInputFieldValue,
  zSDXLRefinerModelInputFieldValue,
  zVaeModelInputFieldValue,
  zLoRAModelInputFieldValue,
  zControlNetModelInputFieldValue,
  zCollectionInputFieldValue,
  zCollectionItemInputFieldValue,
  zColorInputFieldValue,
  zImageCollectionInputFieldValue,
  zSchedulerInputFieldValue,
]);

export type InputFieldValue = z.infer<typeof zInputFieldValue>;

export type InputFieldTemplateBase = {
  name: string;
  title: string;
  description: string;
  type: FieldType;
  required: boolean;
  fieldKind: 'input';
} & _InputField;

export type IntegerInputFieldTemplate = InputFieldTemplateBase & {
  type: 'integer';
  default: number;
  multipleOf?: number;
  maximum?: number;
  exclusiveMaximum?: boolean;
  minimum?: number;
  exclusiveMinimum?: boolean;
};

export type FloatInputFieldTemplate = InputFieldTemplateBase & {
  type: 'float';
  default: number;
  multipleOf?: number;
  maximum?: number;
  exclusiveMaximum?: boolean;
  minimum?: number;
  exclusiveMinimum?: boolean;
};

export type StringInputFieldTemplate = InputFieldTemplateBase & {
  type: 'string';
  default: string;
  maxLength?: number;
  minLength?: number;
  pattern?: string;
};

export type BooleanInputFieldTemplate = InputFieldTemplateBase & {
  default: boolean;
  type: 'boolean';
};

export type ImageInputFieldTemplate = InputFieldTemplateBase & {
  default: ImageDTO;
  type: 'ImageField';
};

export type ImageCollectionInputFieldTemplate = InputFieldTemplateBase & {
  default: ImageField[];
  type: 'ImageCollection';
};

export type DenoiseMaskInputFieldTemplate = InputFieldTemplateBase & {
  default: undefined;
  type: 'DenoiseMaskField';
};

export type LatentsInputFieldTemplate = InputFieldTemplateBase & {
  default: string;
  type: 'LatentsField';
};

export type ConditioningInputFieldTemplate = InputFieldTemplateBase & {
  default: undefined;
  type: 'ConditioningField';
};

export type UNetInputFieldTemplate = InputFieldTemplateBase & {
  default: undefined;
  type: 'UNetField';
};

export type ClipInputFieldTemplate = InputFieldTemplateBase & {
  default: undefined;
  type: 'ClipField';
};

export type VaeInputFieldTemplate = InputFieldTemplateBase & {
  default: undefined;
  type: 'VaeField';
};

export type ControlInputFieldTemplate = InputFieldTemplateBase & {
  default: undefined;
  type: 'ControlField';
};

export type EnumInputFieldTemplate = InputFieldTemplateBase & {
  default: string | number;
  type: 'enum';
  enumType: 'string' | 'number';
  options: Array<string | number>;
};

export type MainModelInputFieldTemplate = InputFieldTemplateBase & {
  default: undefined;
  type: 'MainModelField';
};

export type SDXLMainModelInputFieldTemplate = InputFieldTemplateBase & {
  default: undefined;
  type: 'SDXLMainModelField';
};

export type SDXLRefinerModelInputFieldTemplate = InputFieldTemplateBase & {
  default: undefined;
  type: 'SDXLRefinerModelField';
};

export type VaeModelInputFieldTemplate = InputFieldTemplateBase & {
  default: string;
  type: 'VaeModelField';
};

export type LoRAModelInputFieldTemplate = InputFieldTemplateBase & {
  default: string;
  type: 'LoRAModelField';
};

export type ControlNetModelInputFieldTemplate = InputFieldTemplateBase & {
  default: string;
  type: 'ControlNetModelField';
};

export type CollectionInputFieldTemplate = InputFieldTemplateBase & {
  default: [];
  type: 'Collection';
};

export type CollectionItemInputFieldTemplate = InputFieldTemplateBase & {
  default: undefined;
  type: 'CollectionItem';
};

export type ColorInputFieldTemplate = InputFieldTemplateBase & {
  default: RgbaColor;
  type: 'ColorField';
};

export type SchedulerInputFieldTemplate = InputFieldTemplateBase & {
  default: SchedulerParam;
  type: 'Scheduler';
};

export type WorkflowInputFieldTemplate = InputFieldTemplateBase & {
  default: undefined;
  type: 'WorkflowField';
};

export const isInputFieldValue = (
  field?: InputFieldValue | OutputFieldValue
): field is InputFieldValue => Boolean(field && field.fieldKind === 'input');

export const isInputFieldTemplate = (
  fieldTemplate?: InputFieldTemplate | OutputFieldTemplate
): fieldTemplate is InputFieldTemplate =>
  Boolean(fieldTemplate && fieldTemplate.fieldKind === 'input');

/**
 * JANKY CUSTOMISATION OF OpenAPI SCHEMA TYPES
 */

export type TypeHints = {
  [fieldName: string]: FieldType;
};

export type InvocationSchemaExtra = {
  output: OpenAPIV3.ReferenceObject; // the output of the invocation
  title: string;
  tags?: string[];
  properties: Omit<
    NonNullable<OpenAPIV3.SchemaObject['properties']> &
      (_InputField | _OutputField),
    'type'
  > & {
    type: Omit<OpenAPIV3.SchemaObject, 'default'> & {
      default: AnyInvocationType;
    };
  };
};

export type InvocationSchemaType = {
  default: string; // the type of the invocation
};

export type InvocationBaseSchemaObject = Omit<
  OpenAPIV3.BaseSchemaObject,
  'title' | 'type' | 'properties'
> &
  InvocationSchemaExtra;

export type InvocationOutputSchemaObject = Omit<
  OpenAPIV3.SchemaObject,
  'properties'
> & {
  properties: OpenAPIV3.SchemaObject['properties'] & {
    type: Omit<OpenAPIV3.SchemaObject, 'default'> & {
      default: string;
    };
  } & {
    class: 'output';
  };
};

export type InvocationFieldSchema = OpenAPIV3.SchemaObject & _InputField;

export interface ArraySchemaObject extends InvocationBaseSchemaObject {
  type: OpenAPIV3.ArraySchemaObjectType;
  items: OpenAPIV3.ReferenceObject | OpenAPIV3.SchemaObject;
}
export interface NonArraySchemaObject extends InvocationBaseSchemaObject {
  type?: OpenAPIV3.NonArraySchemaObjectType;
}

export type InvocationSchemaObject = (
  | ArraySchemaObject
  | NonArraySchemaObject
) & { class: 'invocation' };

export const isSchemaObject = (
  obj: OpenAPIV3.ReferenceObject | OpenAPIV3.SchemaObject
): obj is OpenAPIV3.SchemaObject => !('$ref' in obj);

export const isInvocationSchemaObject = (
  obj:
    | OpenAPIV3.ReferenceObject
    | OpenAPIV3.SchemaObject
    | InvocationSchemaObject
): obj is InvocationSchemaObject =>
  'class' in obj && obj.class === 'invocation';

export const isInvocationOutputSchemaObject = (
  obj:
    | OpenAPIV3.ReferenceObject
    | OpenAPIV3.SchemaObject
    | InvocationOutputSchemaObject
): obj is InvocationOutputSchemaObject =>
  'class' in obj && obj.class === 'output';

export const isInvocationFieldSchema = (
  obj: OpenAPIV3.ReferenceObject | OpenAPIV3.SchemaObject
): obj is InvocationFieldSchema => !('$ref' in obj);

export type InvocationEdgeExtra = { type: 'default' | 'collapsed' };

export const zCoreMetadata = z
  .object({
    app_version: z.string().nullish(),
    generation_mode: z.string().nullish(),
    created_by: z.string().nullish(),
    positive_prompt: z.string().nullish(),
    negative_prompt: z.string().nullish(),
    width: z.number().int().nullish(),
    height: z.number().int().nullish(),
    seed: z.number().int().nullish(),
    rand_device: z.string().nullish(),
    cfg_scale: z.number().nullish(),
    steps: z.number().int().nullish(),
    scheduler: z.string().nullish(),
    clip_skip: z.number().int().nullish(),
    model: zMainOrOnnxModel.nullish(),
    controlnets: z.array(zControlField).nullish(),
    loras: z
      .array(
        z.object({
          lora: zLoRAModelField,
          weight: z.number(),
        })
      )
      .nullish(),
    vae: zVaeModelField.nullish(),
    strength: z.number().nullish(),
    init_image: z.string().nullish(),
    positive_style_prompt: z.string().nullish(),
    negative_style_prompt: z.string().nullish(),
    refiner_model: zSDXLRefinerModel.nullish(),
    refiner_cfg_scale: z.number().nullish(),
    refiner_steps: z.number().int().nullish(),
    refiner_scheduler: z.string().nullish(),
    refiner_positive_aesthetic_store: z.number().nullish(),
    refiner_negative_aesthetic_store: z.number().nullish(),
    refiner_start: z.number().nullish(),
  })
  .catchall(z.record(z.any()));

export type CoreMetadata = z.infer<typeof zCoreMetadata>;

export const zInvocationNodeData = z.object({
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
});

// Massage this to get better type safety while developing
export type InvocationNodeData = Omit<
  z.infer<typeof zInvocationNodeData>,
  'type'
> & {
  type: AnyInvocationType;
};

export const zNotesNodeData = z.object({
  id: z.string().trim().min(1),
  type: z.literal('notes'),
  label: z.string(),
  isOpen: z.boolean(),
  notes: z.string(),
});

export type NotesNodeData = z.infer<typeof zNotesNodeData>;

const zPosition = z
  .object({
    x: z.number(),
    y: z.number(),
  })
  .default({ x: 0, y: 0 });

const zDimension = z.number().gt(0).nullish();

export const zWorkflowInvocationNode = z.object({
  id: z.string().trim().min(1),
  type: z.literal('invocation'),
  data: zInvocationNodeData,
  width: zDimension,
  height: zDimension,
  position: zPosition,
});

export type WorkflowInvocationNode = z.infer<typeof zWorkflowInvocationNode>;

export const isWorkflowInvocationNode = (
  val: unknown
): val is WorkflowInvocationNode =>
  zWorkflowInvocationNode.safeParse(val).success;

export const zWorkflowNotesNode = z.object({
  id: z.string().trim().min(1),
  type: z.literal('notes'),
  data: zNotesNodeData,
  width: zDimension,
  height: zDimension,
  position: zPosition,
});

export const zWorkflowNode = z.discriminatedUnion('type', [
  zWorkflowInvocationNode,
  zWorkflowNotesNode,
]);

export type WorkflowNode = z.infer<typeof zWorkflowNode>;

export const zWorkflowEdge = z.object({
  source: z.string().trim().min(1),
  sourceHandle: z.string().trim().min(1),
  target: z.string().trim().min(1),
  targetHandle: z.string().trim().min(1),
  id: z.string().trim().min(1),
  type: z.enum(['default', 'collapsed']),
});

export const zFieldIdentifier = z.object({
  nodeId: z.string().trim().min(1),
  fieldName: z.string().trim().min(1),
});

export type FieldIdentifier = z.infer<typeof zFieldIdentifier>;

export const zSemVer = z.string().refine((val) => {
  const [major, minor, patch] = val.split('.');
  return (
    major !== undefined &&
    minor !== undefined &&
    patch !== undefined &&
    Number.isInteger(Number(major)) &&
    Number.isInteger(Number(minor)) &&
    Number.isInteger(Number(patch))
  );
});

export type SemVer = z.infer<typeof zSemVer>;

export type WorkflowWarning = {
  message: string;
  issues: string[];
  data: JsonObject;
};

export const zWorkflow = z.object({
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
  meta: z
    .object({
      version: zSemVer,
    })
    .default({ version: '1.0.0' }),
});

export const zValidatedWorkflow = zWorkflow.transform((workflow) => {
  const nodeTemplates = store.getState().nodes.nodeTemplates;
  const { nodes, edges } = workflow;
  const warnings: WorkflowWarning[] = [];
  const invocationNodes = nodes.filter(isWorkflowInvocationNode);
  const keyedNodes = keyBy(invocationNodes, 'id');
  invocationNodes.forEach((node, i) => {
    const nodeTemplate = nodeTemplates[node.data.type];
    if (!nodeTemplate) {
      warnings.push({
        message: `Node "${node.data.label || node.data.id}" skipped`,
        issues: [`Unable to find template for type "${node.data.type}"`],
        data: node,
      });
      delete nodes[i];
    }
  });
  edges.forEach((edge, i) => {
    const sourceNode = keyedNodes[edge.source];
    const targetNode = keyedNodes[edge.target];
    const issues: string[] = [];
    if (!sourceNode) {
      issues.push(`Output node ${edge.source} does not exist`);
    } else if (!(edge.sourceHandle in sourceNode.data.outputs)) {
      issues.push(
        `Output field "${edge.source}.${edge.sourceHandle}" does not exist`
      );
    }
    if (!targetNode) {
      issues.push(`Input node ${edge.target} does not exist`);
    } else if (!(edge.targetHandle in targetNode.data.inputs)) {
      issues.push(
        `Input field "${edge.target}.${edge.targetHandle}" does not exist`
      );
    }
    if (issues.length) {
      delete edges[i];
      warnings.push({
        message: `Edge "${edge.sourceHandle} -> ${edge.targetHandle}" skipped`,
        issues,
        data: edge,
      });
    }
  });
  return { workflow, warnings };
});

export type Workflow = z.infer<typeof zWorkflow>;

export type ImageMetadataAndWorkflow = {
  metadata?: CoreMetadata;
  workflow?: Workflow;
};

export type CurrentImageNodeData = {
  id: string;
  type: 'current_image';
  isOpen: boolean;
  label: string;
};

export type NodeData =
  | InvocationNodeData
  | NotesNodeData
  | CurrentImageNodeData;

export const isInvocationNode = (
  node?: Node<NodeData>
): node is Node<InvocationNodeData> =>
  Boolean(node && node.type === 'invocation');

export const isInvocationNodeData = (
  node?: NodeData
): node is InvocationNodeData =>
  Boolean(node && !['notes', 'current_image'].includes(node.type));

export const isNotesNode = (
  node?: Node<NodeData>
): node is Node<NotesNodeData> => Boolean(node && node.type === 'notes');

export const isProgressImageNode = (
  node?: Node<NodeData>
): node is Node<CurrentImageNodeData> =>
  Boolean(node && node.type === 'current_image');

export enum NodeStatus {
  PENDING,
  IN_PROGRESS,
  COMPLETED,
  FAILED,
}

export type NodeExecutionState = {
  nodeId: string;
  status: NodeStatus;
  progress: number | null;
  progressImage: ProgressImage | null;
  error: string | null;
  outputs: AnyResult[];
};

export type FieldComponentProps<
  V extends InputFieldValue,
  T extends InputFieldTemplate,
> = {
  nodeId: string;
  field: V;
  fieldTemplate: T;
};
