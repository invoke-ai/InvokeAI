import { OpenAPIV3 } from 'openapi-types';
import { FunctionComponent } from 'react';

export const isReferenceObject = (
  obj:
    | OpenAPIV3.ReferenceObject
    | OpenAPIV3.SchemaObject
    | NodeSchemaObject
    | ProcessedNodeSchemaObject
): obj is OpenAPIV3.ReferenceObject => '$ref' in obj;

export const isNodeSchemaObject = (
  obj:
    | OpenAPIV3.ReferenceObject
    | OpenAPIV3.SchemaObject
    | NodeSchemaObject
    | ProcessedNodeSchemaObject
): obj is NodeSchemaObject => !('$ref' in obj);

export const isArraySchemaObject = (
  obj: OpenAPIV3.ArraySchemaObject | OpenAPIV3.NonArraySchemaObject
): obj is OpenAPIV3.ArraySchemaObject => 'items' in obj;

export const isNonArraySchemaObject = (
  obj: OpenAPIV3.ArraySchemaObject | OpenAPIV3.NonArraySchemaObject
): obj is OpenAPIV3.NonArraySchemaObject => !('items' in obj);

// helper types - we have some guarantees about the schema - so we can override some optional
// properties

export type RequiredInvocationProperties = {
  type: string;
  title: string;
  id: string;
  output: OpenAPIV3.ReferenceObject; // add the `output` custom schema prop
  properties: OpenAPIV3.ReferenceObject | NodeSchemaObject;
};

export type NodeSchemaObject = Omit<
  OpenAPIV3.SchemaObject,
  keyof RequiredInvocationProperties
> &
  RequiredInvocationProperties;

export type ProcessedNodeSchemaObject = NodeSchemaObject & {
  fieldType: string;
};

export type NodesComponentsObject = Omit<
  OpenAPIV3.ComponentsObject,
  'schemas'
> & {
  // we know we always have schemas
  schemas: {
    [key: string]:
      | OpenAPIV3.ReferenceObject
      | (NodeSchemaObject & { properties: { type: { default: string } } });
  };
};

export type NodesOpenAPIDocument = Omit<OpenAPIV3.Document, 'components'> & {
  // we know we always have components
  components: NodesComponentsObject;
};

export type Invocation = {
  title: string;
  type: string;
  description: string;
  schema: NodeSchemaObject;
  outputs: ProcessedNodeSchemaObject[];
  inputs: ProcessedNodeSchemaObject[];
  component: FunctionComponent;
};

export type Invocations = { [name: string]: Invocation };

export type FieldConfig = {
  [type: string]: {
    color: string;
    isPrimitive: boolean;
  };
};
