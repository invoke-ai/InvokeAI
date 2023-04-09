import { filter, reduce } from 'lodash';
import { OpenAPIV3 } from 'openapi-types';
import {
  InputField,
  Invocation,
  isReferenceObject,
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
  );

  const invocations = filteredSchemas.reduce<Record<string, Invocation>>(
    (acc, schema: OpenAPIV3.ReferenceObject | OpenAPIV3.SchemaObject) => {
      // only want SchemaObjects
      if (isReferenceObject(schema)) {
        return acc;
      }

      const type = (
        schema.properties!.type as OpenAPIV3.SchemaObject & { default: string }
      ).default;

      const title = schema
        .title!.replace('Invocation', '')
        .split(/(?=[A-Z])/) // split PascalCase into array
        .join(' ');

      const inputs = reduce(
        schema.properties,
        (inputsAccumulator, property, propertyName) => {
          if (
            // `type` and `id` are not valid inputs/outputs
            !['type', 'id'].includes(propertyName) &&
            isSchemaObject(property)
          ) {
            const field = buildInputField(property, propertyName);

            if (field) {
              inputsAccumulator[propertyName] = field;
            }
          }
          return inputsAccumulator;
        },
        {} as Record<string, InputField>
      );

      const rawOutput = (
        schema as OpenAPIV3.SchemaObject & {
          output: OpenAPIV3.ReferenceObject;
        }
      ).output;

      const outputs = buildOutputFields(rawOutput, openAPI);

      const invocation: Invocation = {
        title,
        type,
        description: schema.description ?? '',
        inputs,
        outputs,
      };

      acc[type] = invocation;

      return acc;
    },
    {}
  );

  return invocations;
};
