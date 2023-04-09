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
  color: 'red' | 'orange' | 'yellow' | 'green' | 'blue' | 'purple' | 'pink';
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
  | 'latents';

export type InputField =
  | IntegerInputField
  | FloatInputField
  | StringInputField
  | BooleanInputField
  | ImageInputField
  | LatentsInputField
  | EnumInputField;

export type OutputField = FieldBase;

export type FieldBase = {
  name: string;
  title: string;
  description: string;
  type: FieldType;
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
