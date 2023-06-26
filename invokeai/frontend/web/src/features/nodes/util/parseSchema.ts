import { filter, reduce } from 'lodash-es';
import { OpenAPIV3 } from 'openapi-types';
import { isSchemaObject } from '../types/typeGuards';
import {
  InputFieldTemplate,
  InvocationSchemaObject,
  InvocationTemplate,
  OutputFieldTemplate,
} from '../types/types';
import { buildInputFieldTemplate, getFieldType } from './fieldTemplateBuilders';
import { O } from 'ts-toolbelt';

// recursively exclude all properties of type U from T
type DeepExclude<T, U> = T extends U
  ? never
  : T extends object
  ? {
      [K in keyof T]: DeepExclude<T[K], U>;
    }
  : T;

// The schema from swagger-parser is dereferenced, and we know `components` and `components.schemas` exist
type DereferencedOpenAPIDocument = DeepExclude<
  O.Required<OpenAPIV3.Document, 'schemas' | 'components', 'deep'>,
  OpenAPIV3.ReferenceObject
>;

const RESERVED_FIELD_NAMES = ['id', 'type', 'is_intermediate'];

const invocationDenylist = ['Graph', 'InvocationMeta'];

const nodeFilter = (
  schema: DereferencedOpenAPIDocument['components']['schemas'][string],
  key: string
) =>
  key.includes('Invocation') &&
  !key.includes('InvocationOutput') &&
  !invocationDenylist.some((denylistItem) => key.includes(denylistItem));

export const parseSchema = (openAPI: DereferencedOpenAPIDocument) => {
  // filter out non-invocation schemas, plus some tricky invocations for now
  const filteredSchemas = filter(openAPI.components.schemas, nodeFilter);

  const invocations = filteredSchemas.reduce<
    Record<string, InvocationTemplate>
  >((acc, s) => {
    // cast to InvocationSchemaObject, we know the shape
    const schema = s as InvocationSchemaObject;

    const type = schema.properties.type.default;

    const title = schema.ui?.title ?? schema.title.replace('Invocation', '');

    const typeHints = schema.ui?.type_hints;

    const inputs: Record<string, InputFieldTemplate> = {};

    if (type === 'collect') {
      // Special handling for the Collect node
      const itemProperty = schema.properties['item'] as InvocationSchemaObject;
      inputs.item = {
        type: 'item',
        name: 'item',
        description: itemProperty.description ?? '',
        title: 'Collection Item',
        inputKind: 'connection',
        inputRequirement: 'always',
        default: undefined,
      };
    } else if (type === 'iterate') {
      // Special handling for the Iterate node
      const itemProperty = schema.properties[
        'collection'
      ] as InvocationSchemaObject;

      inputs.collection = {
        type: 'array',
        name: 'collection',
        title: itemProperty.title ?? '',
        default: [],
        description: itemProperty.description ?? '',
        inputRequirement: 'always',
        inputKind: 'connection',
      };
    } else {
      // All other nodes
      reduce(
        schema.properties,
        (inputsAccumulator, property, propertyName) => {
          if (
            // `type` and `id` are not valid inputs/outputs
            !RESERVED_FIELD_NAMES.includes(propertyName) &&
            isSchemaObject(property)
          ) {
            const field: InputFieldTemplate | undefined =
              buildInputFieldTemplate(property, propertyName, typeHints);

            if (field) {
              inputsAccumulator[propertyName] = field;
            }
          }
          return inputsAccumulator;
        },
        inputs
      );
    }

    let outputs: Record<string, OutputFieldTemplate>;

    if (type === 'iterate') {
      // Special handling for the Iterate node output
      const iterationOutput =
        openAPI.components.schemas['IterateInvocationOutput'];

      outputs = {
        item: {
          name: 'item',
          title: iterationOutput.title ?? '',
          description: iterationOutput.description ?? '',
          type: 'array',
        },
      };
    } else {
      // All other node outputs
      outputs = reduce(
        schema.output.properties as OpenAPIV3.SchemaObject,
        (outputsAccumulator, property, propertyName) => {
          if (!['type', 'id'].includes(propertyName)) {
            const fieldType = getFieldType(property, propertyName, typeHints);

            outputsAccumulator[propertyName] = {
              name: propertyName,
              title: property.title ?? '',
              description: property.description ?? '',
              type: fieldType,
            };
          }

          return outputsAccumulator;
        },
        {} as Record<string, OutputFieldTemplate>
      );
    }

    const invocation: InvocationTemplate = {
      title,
      type,
      tags: schema.ui?.tags ?? [],
      description: schema.description ?? '',
      inputs,
      outputs,
    };

    Object.assign(acc, { [type]: invocation });

    return acc;
  }, {});

  return invocations;
};
