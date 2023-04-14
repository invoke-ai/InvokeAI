import { OpenAPIV3 } from 'openapi-types';

export const isReferenceObject = (
  obj: OpenAPIV3.ReferenceObject | OpenAPIV3.SchemaObject
): obj is OpenAPIV3.ReferenceObject => '$ref' in obj;

export const isSchemaObject = (
  obj: OpenAPIV3.ReferenceObject | OpenAPIV3.SchemaObject
): obj is OpenAPIV3.SchemaObject => !('$ref' in obj);

export type Invocation = {
  /**
   * Unique type of the invocation
   */
  type: string;
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
  inputs: Record<string, InputField>;
  // inputs: InputField[];
  /**
   * Array of the invocation outputs
   */
  outputs: Record<string, OutputField>;
  // outputs: OutputField[];
};

export type FieldUIConfig = {
  color:
    | 'red'
    | 'orange'
    | 'yellow'
    | 'green'
    | 'blue'
    | 'purple'
    | 'pink'
    | 'teal'
    | 'gray';
  title: string;
  description: string;
};

export type FieldType =
  | 'integer'
  | 'float'
  | 'string'
  | 'boolean'
  | 'enum'
  | 'image'
  | 'latents'
  | 'model'
  | 'array';

export type InputField =
  | IntegerInputField
  | FloatInputField
  | StringInputField
  | BooleanInputField
  | ImageInputField
  | LatentsInputField
  | EnumInputField
  | ModelInputField
  | ArrayInputField;

export type OutputField = FieldBase;

export type ConnectionType = 'never' | 'always';

export type FieldBase = {
  name: string;
  title: string;
  description: string;
  type: FieldType;
  connectionType?: ConnectionType;
};

export type NumberInvocationField = {
  value?: number;
  multipleOf?: number;
  maximum?: number;
  exclusiveMaximum?: boolean;
  minimum?: number;
  exclusiveMinimum?: boolean;
};

export type IntegerInputField = FieldBase &
  NumberInvocationField & {
    type: 'integer';
  };

export type FloatInputField = FieldBase &
  NumberInvocationField & {
    type: 'float';
  };

export type StringInputField = FieldBase & {
  type: 'string';
  value?: string;
  maxLength?: number;
  minLength?: number;
  pattern?: string;
};

export type BooleanInputField = FieldBase & {
  type: 'boolean';
  value?: boolean;
};

export type ImageInputField = FieldBase & {
  type: 'image';
  // TODO: use a better value
  value?: string;
};

export type LatentsInputField = FieldBase & {
  type: 'latents';
  // TODO: use a better value
  value?: string;
};

export type EnumInputField = FieldBase & {
  type: 'enum';
  value?: string | number;
  enumType: 'string' | 'integer' | 'number';
  options: Array<string | number>;
};

export type ModelInputField = FieldBase & {
  type: 'model';
  value?: string;
};

export type ArrayInputField = FieldBase & {
  type: 'array';
  value?: string;
};

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
    type_hints?: TypeHints;
  };
  title: string;
  properties: Omit<
    NonNullable<OpenAPIV3.SchemaObject['properties']>,
    'type'
  > & {
    type: Omit<OpenAPIV3.SchemaObject, 'default'> & { default: string };
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
