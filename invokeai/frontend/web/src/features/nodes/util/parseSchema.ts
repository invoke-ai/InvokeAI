import { filter, reduce } from 'lodash-es';
import { OpenAPIV3 } from 'openapi-types';
import { isSchemaObject } from '../types/typeGuards';
import {
  InputFieldTemplate,
  InvocationSchemaObject,
  InvocationTemplate,
  isInvocationSchemaObject,
  OutputFieldTemplate,
} from '../types/types';
import {
  buildInputFieldTemplate,
  buildOutputFieldTemplates,
} from './fieldTemplateBuilders';

const RESERVED_FIELD_NAMES = ['id', 'type', 'is_intermediate'];

const invocationDenylist = ['Graph', 'InvocationMeta'];

export const parseSchema = (openAPI: OpenAPIV3.Document) => {
  // filter out non-invocation schemas, plus some tricky invocations for now
  const filteredSchemas = filter(
    openAPI.components!.schemas,
    (schema, key) =>
      key.includes('Invocation') &&
      !key.includes('InvocationOutput') &&
      !invocationDenylist.some((denylistItem) => key.includes(denylistItem))
  ) as (OpenAPIV3.ReferenceObject | InvocationSchemaObject)[];

  const invocations = filteredSchemas.reduce<
    Record<string, InvocationTemplate>
  >((acc, schema) => {
    // only want SchemaObjects
    if (isInvocationSchemaObject(schema)) {
      const type = schema.properties.type.default;

      const title = schema.ui?.title ?? schema.title.replace('Invocation', '');

      const typeHints = schema.ui?.type_hints;

      const inputs: Record<string, InputFieldTemplate> = {};

      if (type === 'collect') {
        const itemProperty = schema.properties[
          'item'
        ] as InvocationSchemaObject;
        // Handle the special Collect node
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

      const rawOutput = (schema as InvocationSchemaObject).output;

      let outputs: Record<string, OutputFieldTemplate>;

      // some special handling is needed for collect, iterate and range nodes
      if (type === 'iterate') {
        // this is guaranteed to be a SchemaObject
        const iterationOutput = openAPI.components!.schemas![
          'IterateInvocationOutput'
        ] as OpenAPIV3.SchemaObject;

        outputs = {
          item: {
            name: 'item',
            title: iterationOutput.title ?? '',
            description: iterationOutput.description ?? '',
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
