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

const RESERVED_INPUT_FIELD_NAMES = ['id', 'type', 'metadata'];
const RESERVED_OUTPUT_FIELD_NAMES = ['type'];

const invocationDenylist: AnyInvocationType[] = [
  'graph',
  'metadata_accumulator',
];

const isAllowedInputField = (nodeType: string, fieldName: string) => {
  if (RESERVED_INPUT_FIELD_NAMES.includes(fieldName)) {
    return false;
  }
  if (nodeType === 'collect' && fieldName === 'collection') {
    return false;
  }
  if (nodeType === 'iterate' && fieldName === 'index') {
    return false;
  }
  return true;
};

const isAllowedOutputField = (nodeType: string, fieldName: string) => {
  if (RESERVED_OUTPUT_FIELD_NAMES.includes(fieldName)) {
    return false;
  }
  return true;
};

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
        if (!isAllowedInputField(type, propertyName)) {
          logger('nodes').trace(
            { type, propertyName, property: parseify(property) },
            'Skipped reserved input field'
          );
          return inputsAccumulator;
        }

        if (!isInvocationFieldSchema(property)) {
          logger('nodes').warn(
            { type, propertyName, property: parseify(property) },
            'Unhandled input property'
          );
          return inputsAccumulator;
        }

        const field = buildInputFieldTemplate(schema, property, propertyName);

        if (field) {
          inputsAccumulator[propertyName] = field;
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

    const outputType = outputSchema.properties.type.default;

    const outputs = reduce(
      outputSchema.properties,
      (outputsAccumulator, property, propertyName) => {
        if (!isAllowedOutputField(type, propertyName)) {
          logger('nodes').trace(
            { type, propertyName, property: parseify(property) },
            'Skipped reserved output field'
          );
          return outputsAccumulator;
        }

        if (!isInvocationFieldSchema(property)) {
          logger('nodes').warn(
            { type, propertyName, property: parseify(property) },
            'Unhandled output property'
          );
          return outputsAccumulator;
        }

        const fieldType = getFieldType(property);
        outputsAccumulator[propertyName] = {
          fieldKind: 'output',
          name: propertyName,
          title: property.title ?? '',
          description: property.description ?? '',
          type: fieldType,
          ui_hidden: property.ui_hidden ?? false,
          ui_type: property.ui_type,
          ui_order: property.ui_order,
        };

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
      outputType,
    };

    Object.assign(acc, { [type]: invocation });

    return acc;
  }, {});

  return invocations;
};
