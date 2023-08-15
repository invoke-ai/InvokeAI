import {
  ControlNetModelParam,
  LoRAModelParam,
  MainModelParam,
  OnnxModelParam,
  VaeModelParam,
} from 'features/parameters/types/parameterSchemas';
import { OpenAPIV3 } from 'openapi-types';
import { RgbaColor } from 'react-colorful';
import { Edge, Node } from 'reactflow';
import {
  Graph,
  ImageDTO,
  ImageField,
  _InputField,
  _OutputField,
} from 'services/api/types';
import { AnyInvocationType, ProgressImage } from 'services/events/types';
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
  'FilePath',
  'enum',
  // endregion
]);

export type FieldType = z.infer<typeof zFieldType>;

export const isFieldType = (value: unknown): value is FieldType =>
  zFieldType.safeParse(value).success;

/**
 * An input field is persisted across reloads as part of the user's local state.
 *
 * An input field has three properties:
 * - `id` a unique identifier
 * - `name` the name of the field, which comes from the python dataclass
 * - `value` the field's value
 */
export type InputFieldValue =
  | IntegerInputFieldValue
  | SeedInputFieldValue
  | FloatInputFieldValue
  | StringInputFieldValue
  | BooleanInputFieldValue
  | ImageInputFieldValue
  | LatentsInputFieldValue
  | ConditioningInputFieldValue
  | UNetInputFieldValue
  | ClipInputFieldValue
  | VaeInputFieldValue
  | ControlInputFieldValue
  | EnumInputFieldValue
  | MainModelInputFieldValue
  | SDXLMainModelInputFieldValue
  | SDXLRefinerModelInputFieldValue
  | VaeModelInputFieldValue
  | LoRAModelInputFieldValue
  | ControlNetModelInputFieldValue
  | CollectionInputFieldValue
  | CollectionItemInputFieldValue
  | ColorInputFieldValue
  | ImageCollectionInputFieldValue;

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
  | ImageCollectionInputFieldTemplate;

/**
 * An output field is persisted across as part of the user's local state.
 *
 * An output field has two properties:
 * - `id` a unique identifier
 * - `name` the name of the field, which comes from the python dataclass
 */
export type OutputFieldValue = FieldValueBase & { fieldKind: 'output' };

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
};

/**
 * Indicates the kind of input(s) this field may have.
 */
export type InputKind = 'connection' | 'direct' | 'any';

export type FieldValueBase = {
  id: string;
  name: string;
  type: FieldType;
};

export type InputFieldValueBase = FieldValueBase & {
  fieldKind: 'input';
  label: string;
};

export type IntegerInputFieldValue = InputFieldValueBase & {
  type: 'integer';
  value?: number;
};

export type FloatInputFieldValue = InputFieldValueBase & {
  type: 'float';
  value?: number;
};

export type SeedInputFieldValue = InputFieldValueBase & {
  type: 'Seed';
  value?: number;
};

export type StringInputFieldValue = InputFieldValueBase & {
  type: 'string';
  value?: string;
};

export type BooleanInputFieldValue = InputFieldValueBase & {
  type: 'boolean';
  value?: boolean;
};

export type EnumInputFieldValue = InputFieldValueBase & {
  type: 'enum';
  value?: number | string;
};

export type LatentsInputFieldValue = InputFieldValueBase & {
  type: 'LatentsField';
  value?: undefined;
};

export type ConditioningInputFieldValue = InputFieldValueBase & {
  type: 'ConditioningField';
  value?: string;
};

export type ControlInputFieldValue = InputFieldValueBase & {
  type: 'ControlField';
  value?: undefined;
};

export type UNetInputFieldValue = InputFieldValueBase & {
  type: 'UNetField';
  value?: undefined;
};

export type ClipInputFieldValue = InputFieldValueBase & {
  type: 'ClipField';
  value?: undefined;
};

export type VaeInputFieldValue = InputFieldValueBase & {
  type: 'VaeField';
  value?: undefined;
};

export type ImageInputFieldValue = InputFieldValueBase & {
  type: 'ImageField';
  value?: ImageField;
};

export type ImageCollectionInputFieldValue = InputFieldValueBase & {
  type: 'ImageCollection';
  value?: ImageField[];
};

export type MainModelInputFieldValue = InputFieldValueBase & {
  type: 'MainModelField';
  value?: MainModelParam | OnnxModelParam;
};

export type SDXLMainModelInputFieldValue = InputFieldValueBase & {
  type: 'SDXLMainModelField';
  value?: MainModelParam | OnnxModelParam;
};

export type SDXLRefinerModelInputFieldValue = InputFieldValueBase & {
  type: 'SDXLRefinerModelField';
  value?: MainModelParam | OnnxModelParam;
};

export type VaeModelInputFieldValue = InputFieldValueBase & {
  type: 'VaeModelField';
  value?: VaeModelParam;
};

export type LoRAModelInputFieldValue = InputFieldValueBase & {
  type: 'LoRAModelField';
  value?: LoRAModelParam;
};

export type ControlNetModelInputFieldValue = InputFieldValueBase & {
  type: 'ControlNetModelField';
  value?: ControlNetModelParam;
};

export type CollectionInputFieldValue = InputFieldValueBase & {
  type: 'Collection';
  value?: (string | number)[];
};

export type CollectionItemInputFieldValue = InputFieldValueBase & {
  type: 'CollectionItem';
  value?: undefined;
};

export type ColorInputFieldValue = InputFieldValueBase & {
  type: 'ColorField';
  value?: RgbaColor;
};

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

export const isInputFieldValue = (
  field: InputFieldValue | OutputFieldValue
): field is InputFieldValue => field.fieldKind === 'input';

export const isInputFieldTemplate = (
  fieldTemplate: InputFieldTemplate | OutputFieldTemplate
): fieldTemplate is InputFieldTemplate => fieldTemplate.fieldKind === 'input';

/**
 * JANKY CUSTOMISATION OF OpenAPI SCHEMA TYPES
 */

export type TypeHints = {
  [fieldName: string]: FieldType;
};

export type InvocationSchemaExtra = {
  output: OpenAPIV3.ReferenceObject; // the output of the invocation
  ui?: {
    tags?: string[];
    title?: string;
  };
  title: string;
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

export type InvocationFieldSchema = OpenAPIV3.SchemaObject & _InputField;

export interface ArraySchemaObject extends InvocationBaseSchemaObject {
  type: OpenAPIV3.ArraySchemaObjectType;
  items: OpenAPIV3.ReferenceObject | OpenAPIV3.SchemaObject;
}
export interface NonArraySchemaObject extends InvocationBaseSchemaObject {
  type?: OpenAPIV3.NonArraySchemaObjectType;
}

export type InvocationSchemaObject = ArraySchemaObject | NonArraySchemaObject;

export const isInvocationSchemaObject = (
  obj: OpenAPIV3.ReferenceObject | InvocationSchemaObject
): obj is InvocationSchemaObject => !('$ref' in obj);

export const isInvocationFieldSchema = (
  obj: OpenAPIV3.ReferenceObject | OpenAPIV3.SchemaObject
): obj is InvocationFieldSchema => !('$ref' in obj);

export type InvocationEdgeExtra = { type: 'default' | 'collapsed' };

export const zInputFieldValue = z.object({
  id: z.string().trim().min(1),
  name: z.string().trim().min(1),
  type: zFieldType,
  label: z.string(),
  isExposed: z.boolean(),
});

export const zInvocationNodeData = z.object({
  id: z.string().trim().min(1),
  type: z.string().trim().min(1),
  inputs: z.record(z.any()),
  outputs: z.record(z.any()),
  label: z.string(),
  isOpen: z.boolean(),
  notes: z.string(),
});

export const zNotesNodeData = z.object({
  id: z.string().trim().min(1),
  type: z.literal('notes'),
  label: z.string(),
  isOpen: z.boolean(),
  notes: z.string(),
});

export const zWorkflow = z.object({
  name: z.string().trim().min(1),
  author: z.string(),
  description: z.string(),
  version: z.string(),
  contact: z.string(),
  tags: z.string(),
  notes: z.string(),
  nodes: z.array(
    z.object({
      id: z.string().trim().min(1),
      type: z.string().trim().min(1),
      data: z.union([zInvocationNodeData, zNotesNodeData]),
      width: z.number().gt(0),
      height: z.number().gt(0),
      position: z.object({
        x: z.number(),
        y: z.number(),
      }),
    })
  ),
  edges: z.array(
    z.object({
      source: z.string().trim().min(1),
      sourceHandle: z.string().trim().min(1),
      target: z.string().trim().min(1),
      targetHandle: z.string().trim().min(1),
      id: z.string().trim().min(1),
      type: z.string().trim().min(1),
    })
  ),
});

export type Workflow = {
  name: string;
  author: string;
  description: string;
  version: string;
  contact: string;
  tags: string;
  notes: string;
  nodes: Pick<
    Node<InvocationNodeData | NotesNodeData>,
    'id' | 'type' | 'data' | 'width' | 'height' | 'position'
  >[];
  edges: Pick<
    Edge<InvocationEdgeExtra>,
    'source' | 'sourceHandle' | 'target' | 'targetHandle' | 'id' | 'type'
  >[];
  exposedFields: FieldIdentifier[];
};

export type InvocationNodeData = {
  id: string;
  type: AnyInvocationType;
  inputs: Record<string, InputFieldValue>;
  outputs: Record<string, OutputFieldValue>;
  label: string;
  isOpen: boolean;
  notes: string;
};

export type NotesNodeData = {
  id: string;
  type: 'notes';
  label: string;
  notes: string;
  isOpen: boolean;
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
): node is Node<InvocationNodeData> => node?.type === 'invocation';

export const isInvocationNodeData = (
  node?: NodeData
): node is InvocationNodeData =>
  !['notes', 'current_image'].includes(node?.type ?? '');

export const isNotesNode = (
  node?: Node<NodeData>
): node is Node<NotesNodeData> => node?.type === 'notes';

export const isProgressImageNode = (
  node?: Node<NodeData>
): node is Node<CurrentImageNodeData> => node?.type === 'current_image';

export enum NodeStatus {
  PENDING,
  IN_PROGRESS,
  COMPLETED,
  FAILED,
}

export type NodeExecutionState = {
  status: NodeStatus;
  progress: number | null;
  progressImage: ProgressImage | null;
  error: string | null;
};

export type FieldIdentifier = {
  nodeId: string;
  fieldName: string;
};
