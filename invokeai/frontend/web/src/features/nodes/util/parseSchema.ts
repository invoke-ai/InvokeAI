import { logger } from 'app/logging/logger';
import { parseify } from 'common/util/serialize';
import { reduce, startCase } from 'lodash-es';
import { OpenAPIV3_1 } from 'openapi-types';
import { AnyInvocationType } from 'services/events/types';
import {
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

const RESERVED_INPUT_FIELD_NAMES = ['id', 'type', 'use_cache'];
const RESERVED_OUTPUT_FIELD_NAMES = ['type'];
const RESERVED_FIELD_TYPES = ['IsIntermediate'];

const invocationDenylist: AnyInvocationType[] = ['graph', 'linear_ui_output'];

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

const isReservedFieldType = (fieldType: string) => {
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
  openAPI: OpenAPIV3_1.Document,
  nodesAllowlistExtra: string[] | undefined = undefined,
  nodesDenylistExtra: string[] | undefined = undefined
): Record<string, InvocationTemplate> => {
  const filteredSchemas = Object.values(openAPI.components?.schemas ?? {})
    .filter(isInvocationSchemaObject)
    .filter(isNotInDenylist)
    .filter((schema) =>
      nodesAllowlistExtra
        ? nodesAllowlistExtra.includes(schema.properties.type.default)
        : true
    )
    .filter((schema) =>
      nodesDenylistExtra
        ? !nodesDenylistExtra.includes(schema.properties.type.default)
        : true
    );

  const invocations = filteredSchemas.reduce<
    Record<string, InvocationTemplate>
  >((invocationsAccumulator, schema) => {
    const type = schema.properties.type.default;
    const title = schema.title.replace('Invocation', '');
    const tags = schema.tags ?? [];
    const description = schema.description ?? '';
    const version = schema.version;
    let withWorkflow = false;

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

        const fieldTypeResult = property.ui_type
          ? { type: property.ui_type, originalType: property.ui_type }
          : getFieldType(property);

        if (!fieldTypeResult) {
          logger('nodes').warn(
            {
              node: type,
              fieldName: propertyName,
              field: parseify(property),
            },
            'Missing input field type'
          );
          return inputsAccumulator;
        }

        // stash this for custom types
        const { type: fieldType, originalType } = fieldTypeResult;

        if (fieldType === 'WorkflowField') {
          withWorkflow = true;
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
            `Skipping reserved input field type: ${fieldType}`
          );
          return inputsAccumulator;
        }

        if (!isFieldType(originalType)) {
          logger('nodes').debug(
            {
              node: type,
              fieldName: propertyName,
              fieldType,
              field: parseify(property),
            },
            `Fallback handling for unknown input field type: ${fieldType}`
          );
        }

        if (!isFieldType(fieldType)) {
          logger('nodes').warn(
            {
              node: type,
              fieldName: propertyName,
              fieldType,
              field: parseify(property),
            },
            `Unable to parse field type: ${fieldType}`
          );
          return inputsAccumulator;
        }

        const field = buildInputFieldTemplate(
          schema,
          property,
          propertyName,
          fieldType,
          originalType
        );

        if (!field) {
          logger('nodes').warn(
            {
              node: type,
              fieldName: propertyName,
              fieldType,
              originalType,
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

        const fieldTypeResult = property.ui_type
          ? { type: property.ui_type, originalType: property.ui_type }
          : getFieldType(property);

        if (!fieldTypeResult) {
          logger('nodes').warn(
            {
              node: type,
              fieldName: propertyName,
              field: parseify(property),
            },
            'Missing output field type'
          );
          return outputsAccumulator;
        }

        const { type: fieldType, originalType } = fieldTypeResult;

        if (!isFieldType(fieldType)) {
          logger('nodes').debug(
            {
              node: type,
              fieldName: propertyName,
              fieldType,
              originalType,
              field: parseify(property),
            },
            `Fallback handling for unknown output field type: ${fieldType}`
          );
        }

        if (!isFieldType(fieldType)) {
          logger('nodes').warn(
            {
              node: type,
              fieldName: propertyName,
              fieldType,
              field: parseify(property),
            },
            `Unable to parse field type: ${fieldType}`
          );
          return outputsAccumulator;
        }

        outputsAccumulator[propertyName] = {
          fieldKind: 'output',
          name: propertyName,
          title:
            property.title ?? (propertyName ? startCase(propertyName) : ''),
          description: property.description ?? '',
          type: fieldType,
          ui_hidden: property.ui_hidden ?? false,
          ui_type: property.ui_type,
          ui_order: property.ui_order,
          originalType,
        };

        return outputsAccumulator;
      },
      {} as Record<string, OutputFieldTemplate>
    );

    const useCache = schema.properties.use_cache.default;

    const invocation: InvocationTemplate = {
      title,
      type,
      version,
      tags,
      description,
      outputType,
      inputs,
      outputs,
      useCache,
      withWorkflow,
    };

    Object.assign(invocationsAccumulator, { [type]: invocation });

    return invocationsAccumulator;
  }, {});

  return invocations;
};
