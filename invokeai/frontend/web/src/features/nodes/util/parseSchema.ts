import { filter } from 'lodash';
import { OpenAPIV3 } from 'openapi-types';
import {
  InputField,
  _Invocation,
  _isReferenceObject,
  _isSchemaObject,
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

  const invocations = filteredSchemas.reduce<Record<string, _Invocation>>(
    (acc, schema: OpenAPIV3.ReferenceObject | OpenAPIV3.SchemaObject) => {
      // only want SchemaObjects
      if (_isReferenceObject(schema)) {
        return acc;
      }

      const type = (
        schema.properties!.type as OpenAPIV3.SchemaObject & { default: string }
      ).default;

      const title = schema
        .title!.replace('Invocation', '')
        .split(/(?=[A-Z])/) // split PascalCase into array
        .join(' ');

      // `type` and `id` are not valid inputs/outputs
      const rawInputs = filter(
        schema.properties,
        (prop, key) => !['type', 'id'].includes(key) && _isSchemaObject(prop)
      ) as OpenAPIV3.SchemaObject[];

      const inputs: InputField[] = [];

      rawInputs.forEach((input) => {
        const field = buildInputField(input);
        if (field) {
          inputs.push(field);
        }
      });

      // `type` and `id` are not valid inputs/outputs
      const rawOutputs = (
        schema as OpenAPIV3.SchemaObject & {
          output: OpenAPIV3.ReferenceObject;
        }
      ).output;

      const outputs = buildOutputFields(rawOutputs, openAPI);

      const invocation: _Invocation = {
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
