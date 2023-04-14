import { filter, reduce } from 'lodash';
import { OpenAPIV3 } from 'openapi-types';
import {
  FieldBase,
  InputField,
  Invocation,
  InvocationBaseSchemaObject,
  InvocationSchemaObject,
  isInvocationSchemaObject,
  isSchemaObject,
  NonArraySchemaObject,
  OutputField,
} from '../types';
import { buildInputField, buildOutputFields } from './invocationFieldBuilders';

const invocationBlacklist = ['Graph', 'Collect', 'LoadImage'];

export const parseSchema = (openAPI: OpenAPIV3.Document) => {
  // filter out non-invocation schemas, plus some tricky invocations for now
  const filteredSchemas = filter(
    openAPI.components!.schemas,
    (schema, key) =>
      key.includes('Invocation') &&
      !key.includes('InvocationOutput') &&
      !invocationBlacklist.some((blacklistItem) => key.includes(blacklistItem))
  ) as (OpenAPIV3.ReferenceObject | InvocationSchemaObject)[];

  const invocations = filteredSchemas.reduce<Record<string, Invocation>>(
    (acc, schema) => {
      // only want SchemaObjects
      if (isInvocationSchemaObject(schema)) {
        const type = schema.properties.type.default;

        const title = schema.title
          .replace('Invocation', '')
          .split(/(?=[A-Z])/) // split PascalCase into array
          .join(' ');

        const typeHints = schema.ui?.type_hints;

        const inputs = reduce(
          schema.properties,
          (inputsAccumulator, property, propertyName) => {
            if (
              // `type` and `id` are not valid inputs/outputs
              !['type', 'id'].includes(propertyName) &&
              isSchemaObject(property)
            ) {
              let field: InputField | undefined;
              if (propertyName === 'collection') {
                field = {
                  name: 'collection',
                  title: property.title ?? '',
                  description: property.description ?? '',
                  type: 'array',
                  connectionType: 'always',
                };
              } else {
                field = buildInputField(property, propertyName, typeHints);
              }
              if (field) {
                inputsAccumulator[propertyName] = field;
              }
            }
            return inputsAccumulator;
          },
          {} as Record<string, InputField>
        );

        const rawOutput = (schema as InvocationSchemaObject).output;

        let outputs: Record<string, OutputField>;

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
              connectionType: 'always',
            },
          };
        } else {
          outputs = buildOutputFields(rawOutput, openAPI, typeHints);
        }

        const invocation: Invocation = {
          title,
          type,
          tags: schema.ui?.tags ?? [],
          description: schema.description ?? '',
          inputs,
          outputs,
        };

        acc[type] = invocation;
      }

      return acc;
    },
    {}
  );

  console.debug('Generated invocations: ', invocations);

  return invocations;
};
