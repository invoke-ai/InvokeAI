import { filter, reduce } from 'lodash-es';
import { OpenAPIV3 } from 'openapi-types';
import {
  InputFieldTemplate,
  InvocationSchemaObject,
  InvocationTemplate,
  isInvocationFieldSchema,
  isInvocationSchemaObject,
} from '../types/types';
import {
  buildInputFieldTemplate,
  buildOutputFieldTemplates,
} from './fieldTemplateBuilders';

const RESERVED_FIELD_NAMES = ['id', 'type', 'metadata'];

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
      const title = schema.ui?.title ?? schema.title.replace('Invocation', '');
      const tags = schema.ui?.tags ?? [];
      const description = schema.description ?? '';

      const inputs = reduce(
        schema.properties,
        (inputsAccumulator, property, propertyName) => {
          if (
            !RESERVED_FIELD_NAMES.includes(propertyName) &&
            isInvocationFieldSchema(property) &&
            !property.ui_hidden
          ) {
            const field = buildInputFieldTemplate(
              schema,
              property,
              propertyName
            );

            if (field) {
              inputsAccumulator[propertyName] = field;
            }
          }
          return inputsAccumulator;
        },
        {} as Record<string, InputFieldTemplate>
      );

      const rawOutput = (schema as InvocationSchemaObject).output;
      const outputs = buildOutputFieldTemplates(rawOutput, openAPI);

      const invocation: InvocationTemplate = {
        title,
        type,
        tags,
        description,
        inputs,
        outputs,
      };

      Object.assign(acc, { [type]: invocation });
    }

    return acc;
  }, {});

  return invocations;
};
