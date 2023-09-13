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
import i18n from 'i18next';

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
  openAPI: OpenAPIV3.Document,
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
            i18n.t('nodes.skippedReservedInput')
          );
          return inputsAccumulator;
        }

        if (!isInvocationFieldSchema(property)) {
          logger('nodes').warn(
            { node: type, propertyName, property: parseify(property) },
            i18n.t('nodes.unhandledInputProperty')
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
            i18n.t('nodes.skippingUnknownInputType')
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
            i18n.t('nodes.skippingReservedFieldType')
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
            i18n.t('nodes.skippingInputNoTemplate')
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
        i18n.t('nodes.noOutputSchemaName')
      );
      return invocationsAccumulator;
    }

    const outputSchema = openAPI.components?.schemas?.[outputSchemaName];
    if (!outputSchema) {
      logger('nodes').warn(
        { outputSchemaName },
        i18n.t('nodes.outputSchemaNotFound')
      );
      return invocationsAccumulator;
    }

    if (!isInvocationOutputSchemaObject(outputSchema)) {
      logger('nodes').error(
        { outputSchema: parseify(outputSchema) },
        i18n.t('nodes.invalidOutputSchema')
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
            i18n.t('nodes.skippedReservedOutput')
          );
          return outputsAccumulator;
        }

        if (!isInvocationFieldSchema(property)) {
          logger('nodes').warn(
            { type, propertyName, property: parseify(property) },
            i18n.t('nodes.unhandledOutputProperty')
          );
          return outputsAccumulator;
        }

        const fieldType = getFieldType(property);

        if (!isFieldType(fieldType)) {
          logger('nodes').warn(
            { fieldName: propertyName, fieldType, field: parseify(property) },
            i18n.t('nodes.skippingUnknownOutputType')
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
