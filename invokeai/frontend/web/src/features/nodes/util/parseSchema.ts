import { filter, reduce } from 'lodash-es';
import { OpenAPIV3 } from 'openapi-types';
import { isSchemaObject } from '../types/typeGuards';
import {
  InputFieldTemplate,
  InvocationSchemaObject,
  InvocationTemplate,
  OutputFieldTemplate,
  isInvocationSchemaObject,
} from '../types/types';
import {
  buildInputFieldTemplate,
  buildOutputFieldTemplates,
} from './fieldTemplateBuilders';

const getReservedFieldNames = (type: string): string[] => {
  if (type === 'l2i') {
    return ['id', 'type', 'metadata'];
  }
  return ['id', 'type', 'is_intermediate', 'metadata'];
};

const invocationDenylist = [
  'Graph',
  'InvocationMeta',
  'MetadataAccumulatorInvocation',
];

export const parseSchema = (
  openAPI: OpenAPIV3.Document
): Record<string, InvocationTemplate> => {
  const filteredSchemas = filter(
    openAPI.components?.schemas,
    (schema, key) =>
      key.includes('Invocation') &&
      !key.includes('InvocationOutput') &&
      !invocationDenylist.some((denylistItem) => key.includes(denylistItem))
  ) as (OpenAPIV3.ReferenceObject | InvocationSchemaObject)[];

  const invocations = filteredSchemas.reduce<
    Record<string, InvocationTemplate>
  >((acc, schema) => {
    if (isInvocationSchemaObject(schema)) {
      const type = schema.properties.type.default;
      const RESERVED_FIELD_NAMES = getReservedFieldNames(type);

      const title = schema.ui?.title ?? schema.title.replace('Invocation', '');
      const typeHints = schema.ui?.type_hints;

      const inputs: Record<string, InputFieldTemplate> = {};

      if (type === 'collect') {
        const itemProperty = schema.properties.item as InvocationSchemaObject;
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
        const itemProperty = schema.properties
          .collection as InvocationSchemaObject;
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
        reduce(
          schema.properties,
          (inputsAccumulator, property, propertyName) => {
            if (
              !RESERVED_FIELD_NAMES.includes(propertyName) &&
              isSchemaObject(property)
            ) {
              const field = buildInputFieldTemplate(
                property,
                propertyName,
                typeHints
              );
              if (field) {
                inputsAccumulator[propertyName] = field;
              }
            }
            return inputsAccumulator;
          },
          inputs
        );
      }

      const rawOutput = (schema as InvocationSchemaObject).output;
      let outputs: Record<string, OutputFieldTemplate>;

      if (type === 'iterate') {
        const iterationOutput = openAPI.components?.schemas?.[
          'IterateInvocationOutput'
        ] as OpenAPIV3.SchemaObject;
        outputs = {
          item: {
            name: 'item',
            title: iterationOutput?.title ?? '',
            description: iterationOutput?.description ?? '',
            type: 'array',
          },
        };
      } else {
        outputs = buildOutputFieldTemplates(rawOutput, openAPI, typeHints);
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
    }

    return acc;
  }, {});

  return invocations;
};
