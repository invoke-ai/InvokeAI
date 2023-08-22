import { logger } from 'app/logging/logger';
import { parseify } from 'common/util/serialize';
import { reduce } from 'lodash-es';
import { OpenAPIV3 } from 'openapi-types';
import { AnyInvocationType } from 'services/events/types';
import {
  InputFieldTemplate,
  InvocationSchemaObject,
  InvocationTemplate,
  OutputFieldTemplate,
  isInvocationFieldSchema,
  isInvocationOutputSchemaObject,
  isInvocationSchemaObject,
} from '../types/types';
import { buildInputFieldTemplate, getFieldType } from './fieldTemplateBuilders';

const RESERVED_FIELD_NAMES = ['id', 'type', 'metadata'];

const invocationDenylist: AnyInvocationType[] = [
  'graph',
  'metadata_accumulator',
];

const isNotInDenylist = (schema: InvocationSchemaObject) =>
  !invocationDenylist.includes(schema.properties.type.default);

export const parseSchema = (
  openAPI: OpenAPIV3.Document
): Record<string, InvocationTemplate> => {
  const filteredSchemas = Object.values(openAPI.components?.schemas ?? {})
    .filter(isInvocationSchemaObject)
    .filter(isNotInDenylist);

  const invocations = filteredSchemas.reduce<
    Record<string, InvocationTemplate>
  >((acc, schema) => {
    const type = schema.properties.type.default;
    const title = schema.title.replace('Invocation', '');
    const tags = schema.tags ?? [];
    const description = schema.description ?? '';

    const inputs = reduce(
      schema.properties,
      (inputsAccumulator, property, propertyName) => {
        if (
          !RESERVED_FIELD_NAMES.includes(propertyName) &&
          isInvocationFieldSchema(property) &&
          !property.ui_hidden
        ) {
          const field = buildInputFieldTemplate(schema, property, propertyName);

          if (field) {
            inputsAccumulator[propertyName] = field;
          }
        }
        return inputsAccumulator;
      },
      {} as Record<string, InputFieldTemplate>
    );

    const outputSchemaName = schema.output.$ref.split('/').pop();

    if (!outputSchemaName) {
      logger('nodes').error(
        { outputRefObject: parseify(schema.output) },
        'No output schema name found in ref object'
      );
      throw 'No output schema name found in ref object';
    }

    const outputSchema = openAPI.components?.schemas?.[outputSchemaName];
    if (!outputSchema) {
      logger('nodes').error({ outputSchemaName }, 'Output schema not found');
      throw 'Output schema not found';
    }

    if (!isInvocationOutputSchemaObject(outputSchema)) {
      logger('nodes').error(
        { outputSchema: parseify(outputSchema) },
        'Invalid output schema'
      );
      throw 'Invalid output schema';
    }

    const outputs = reduce(
      outputSchema.properties as OpenAPIV3.SchemaObject,
      (outputsAccumulator, property, propertyName) => {
        if (
          !['type', 'id'].includes(propertyName) &&
          !['object'].includes(property.type) && // TODO: handle objects?
          isInvocationFieldSchema(property)
        ) {
          const fieldType = getFieldType(property);
          outputsAccumulator[propertyName] = {
            fieldKind: 'output',
            name: propertyName,
            title: property.title ?? '',
            description: property.description ?? '',
            type: fieldType,
          };
        } else {
          logger('nodes').warn({ property }, 'Unhandled output property');
        }

        return outputsAccumulator;
      },
      {} as Record<string, OutputFieldTemplate>
    );

    const invocation: InvocationTemplate = {
      title,
      type,
      tags,
      description,
      inputs,
      outputs,
    };

    Object.assign(acc, { [type]: invocation });

    return acc;
  }, {});

  return invocations;
};
