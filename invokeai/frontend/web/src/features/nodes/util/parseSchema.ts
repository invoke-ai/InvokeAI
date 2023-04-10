import { filter, reduce } from 'lodash';
import { OpenAPIV3 } from 'openapi-types';
import {
  InputField,
  Invocation,
  InvocationSchemaObject,
  isInvocationSchemaObject,
  isSchemaObject,
} from '../types';
import { buildInputField, buildOutputFields } from './invocationFieldBuilders';

export const parseSchema = (openAPI: OpenAPIV3.Document) => {
  // filter out non-invocation schemas, plus some tricky invocations for now
  const filteredSchemas = filter(
    openAPI.components!.schemas,
    (schema, key) =>
      key.includes('Invocation') &&
      !key.includes('InvocationOutput') &&
      !key.includes('Collect') &&
      !key.includes('Range') &&
      !key.includes('Iterate') &&
      !key.includes('LoadImage') &&
      !key.includes('Graph')
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
              const field = buildInputField(property, propertyName, typeHints);

              if (field) {
                inputsAccumulator[propertyName] = field;
              }
            }
            return inputsAccumulator;
          },
          {} as Record<string, InputField>
        );

        const rawOutput = (schema as InvocationSchemaObject).output;

        const outputs = buildOutputFields(rawOutput, openAPI, typeHints);

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
