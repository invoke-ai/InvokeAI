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

const invocationDenylist = ['Graph', 'Collect', 'LoadImage'];

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

      const title =
        schema.ui?.title ??
        schema.title
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
            let field: InputFieldTemplate | undefined;
            if (propertyName === 'collection') {
              field = {
                default: property.default ?? [],
                name: 'collection',
                title: property.title ?? '',
                description: property.description ?? '',
                type: 'array',
                inputRequirement: 'always',
                inputKind: 'connection',
              };
            } else {
              field = buildInputFieldTemplate(
                property,
                propertyName,
                typeHints
              );
            }
            if (field) {
              inputsAccumulator[propertyName] = field;
            }
          }
          return inputsAccumulator;
        },
        {} as Record<string, InputFieldTemplate>
      );

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
