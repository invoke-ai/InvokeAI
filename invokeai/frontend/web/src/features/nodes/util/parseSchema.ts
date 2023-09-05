import { logger } from 'app/logging/logger';
import { parseify } from 'common/util/serialize';
import { reduce } from 'lodash-es';
import { OpenAPIV3 } from 'openapi-types';
import { AnyInvocationType } from 'services/events/types';
import {
  FieldType,
  InputFieldTemplate,
  InvocationSchemaObject,
  InvocationTemplate,
  OutputFieldTemplate,
  isFieldType,
  isInvocationFieldSchema,
  isInvocationOutputSchemaObject,
  isInvocationSchemaObject,
} from '../types/types';
import { buildInputFieldTemplate, getFieldType } from './fieldTemplateBuilders';

const RESERVED_INPUT_FIELD_NAMES = ['id', 'type', 'metadata'];
const RESERVED_OUTPUT_FIELD_NAMES = ['type'];
const RESERVED_FIELD_TYPES = [
  'WorkflowField',
  'MetadataField',
  'IsIntermediate',
];

const invocationDenylist: AnyInvocationType[] = [
  'graph',
  'metadata_accumulator',
];

const isReservedInputField = (nodeType: string, fieldName: string) => {
  if (RESERVED_INPUT_FIELD_NAMES.includes(fieldName)) {
    return true;
  }
  if (nodeType === 'collect' && fieldName === 'collection') {
    return true;
  }
  if (nodeType === 'iterate' && fieldName === 'index') {
    return true;
  }
  return false;
};

const isReservedFieldType = (fieldType: FieldType) => {
  if (RESERVED_FIELD_TYPES.includes(fieldType)) {
    return true;
  }
  return false;
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
  >((invocationsAccumulator, schema) => {
    const type = schema.properties.type.default;
    const title = schema.title.replace('Invocation', '');
    const tags = schema.tags ?? [];
    const description = schema.description ?? '';
    const version = schema.version ?? '';

    const inputs = reduce(
      schema.properties,
      (
        inputsAccumulator: Record<string, InputFieldTemplate>,
        property,
        propertyName
      ) => {
        if (isReservedInputField(type, propertyName)) {
          logger('nodes').trace(
            { node: type, fieldName: propertyName, field: parseify(property) },
            'Skipped reserved input field'
          );
          return inputsAccumulator;
        }

        if (!isInvocationFieldSchema(property)) {
          logger('nodes').warn(
            { node: type, propertyName, property: parseify(property) },
            'Unhandled input property'
          );
          return inputsAccumulator;
        }

        const fieldType = getFieldType(property);

        if (!isFieldType(fieldType)) {
          logger('nodes').warn(
            {
              node: type,
              fieldName: propertyName,
              fieldType,
              field: parseify(property),
            },
            'Skipping unknown input field type'
          );
          return inputsAccumulator;
        }

        if (isReservedFieldType(fieldType)) {
          logger('nodes').trace(
            {
              node: type,
              fieldName: propertyName,
              fieldType,
              field: parseify(property),
            },
            'Skipping reserved field type'
          );
          return inputsAccumulator;
        }

        const field = buildInputFieldTemplate(
          schema,
          property,
          propertyName,
          fieldType
        );

        if (!field) {
          logger('nodes').debug(
            {
              node: type,
              fieldName: propertyName,
              fieldType,
              field: parseify(property),
            },
            'Skipping input field with no template'
          );
          return inputsAccumulator;
        }

        inputsAccumulator[propertyName] = field;
        return inputsAccumulator;
      },
      {}
    );

    const outputSchemaName = schema.output.$ref.split('/').pop();

    if (!outputSchemaName) {
      logger('nodes').warn(
        { outputRefObject: parseify(schema.output) },
        'No output schema name found in ref object'
      );
      return invocationsAccumulator;
    }

    const outputSchema = openAPI.components?.schemas?.[outputSchemaName];
    if (!outputSchema) {
      logger('nodes').warn({ outputSchemaName }, 'Output schema not found');
      return invocationsAccumulator;
    }

    if (!isInvocationOutputSchemaObject(outputSchema)) {
      logger('nodes').error(
        { outputSchema: parseify(outputSchema) },
        'Invalid output schema'
      );
      return invocationsAccumulator;
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

        if (!isFieldType(fieldType)) {
          logger('nodes').warn(
            { fieldName: propertyName, fieldType, field: parseify(property) },
            'Skipping unknown output field type'
          );
          return outputsAccumulator;
        }

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
      version,
      tags,
      description,
      outputType,
      inputs,
      outputs,
    };

    Object.assign(invocationsAccumulator, { [type]: invocation });

    return invocationsAccumulator;
  }, {});

  return invocations;
};
