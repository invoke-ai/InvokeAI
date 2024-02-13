import type { OpenAPIV3_1 } from 'openapi-types';
import type {
  InputFieldJSONSchemaExtra,
  InvocationJSONSchemaExtra,
  OutputFieldJSONSchemaExtra,
} from 'services/api/types';

// Janky customization of OpenAPI Schema :/

export type InvocationSchemaExtra = InvocationJSONSchemaExtra & {
  output: OpenAPIV3_1.ReferenceObject; // the output of the invocation
  title: string;
  category?: string;
  tags?: string[];
  version: string;
  properties: Omit<
    NonNullable<OpenAPIV3_1.SchemaObject['properties']> & (InputFieldJSONSchemaExtra | OutputFieldJSONSchemaExtra),
    'type'
  > & {
    type: Omit<OpenAPIV3_1.SchemaObject, 'default'> & {
      default: string;
    };
    use_cache: Omit<OpenAPIV3_1.SchemaObject, 'default'> & {
      default: boolean;
    };
  };
};

export type InvocationSchemaType = {
  default: string; // the type of the invocation
};

export type InvocationBaseSchemaObject = Omit<OpenAPIV3_1.BaseSchemaObject, 'title' | 'type' | 'properties'> &
  InvocationSchemaExtra;

export type InvocationOutputSchemaObject = Omit<OpenAPIV3_1.SchemaObject, 'properties'> & {
  properties: OpenAPIV3_1.SchemaObject['properties'] & {
    type: Omit<OpenAPIV3_1.SchemaObject, 'default'> & {
      default: string;
    };
  } & {
    class: 'output';
  };
};

export type InvocationFieldSchema = OpenAPIV3_1.SchemaObject & InputFieldJSONSchemaExtra;

export type OpenAPIV3_1SchemaOrRef = OpenAPIV3_1.ReferenceObject | OpenAPIV3_1.SchemaObject;

export interface ArraySchemaObject extends InvocationBaseSchemaObject {
  type: OpenAPIV3_1.ArraySchemaObjectType;
  items: OpenAPIV3_1.ReferenceObject | OpenAPIV3_1.SchemaObject;
}
export interface NonArraySchemaObject extends InvocationBaseSchemaObject {
  type?: OpenAPIV3_1.NonArraySchemaObjectType;
}

export type InvocationSchemaObject = (ArraySchemaObject | NonArraySchemaObject) & { class: 'invocation' };

export const isSchemaObject = (
  obj: OpenAPIV3_1.ReferenceObject | OpenAPIV3_1.SchemaObject | undefined
): obj is OpenAPIV3_1.SchemaObject => Boolean(obj && !('$ref' in obj));

export const isArraySchemaObject = (
  obj: OpenAPIV3_1.ReferenceObject | OpenAPIV3_1.SchemaObject | undefined
): obj is OpenAPIV3_1.ArraySchemaObject => Boolean(obj && !('$ref' in obj) && obj.type === 'array');

export const isNonArraySchemaObject = (
  obj: OpenAPIV3_1.ReferenceObject | OpenAPIV3_1.SchemaObject | undefined
): obj is OpenAPIV3_1.NonArraySchemaObject => Boolean(obj && !('$ref' in obj) && obj.type !== 'array');

export const isRefObject = (
  obj: OpenAPIV3_1.ReferenceObject | OpenAPIV3_1.SchemaObject | undefined
): obj is OpenAPIV3_1.ReferenceObject => Boolean(obj && '$ref' in obj);

export const isInvocationSchemaObject = (
  obj: OpenAPIV3_1.ReferenceObject | OpenAPIV3_1.SchemaObject | InvocationSchemaObject
): obj is InvocationSchemaObject => 'class' in obj && obj.class === 'invocation';

export const isInvocationOutputSchemaObject = (
  obj: OpenAPIV3_1.ReferenceObject | OpenAPIV3_1.SchemaObject | InvocationOutputSchemaObject
): obj is InvocationOutputSchemaObject => 'class' in obj && obj.class === 'output';

export const isInvocationFieldSchema = (
  obj: OpenAPIV3_1.ReferenceObject | OpenAPIV3_1.SchemaObject
): obj is InvocationFieldSchema => !('$ref' in obj);
