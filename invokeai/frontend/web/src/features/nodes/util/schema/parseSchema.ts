import { logger } from 'app/logging/logger';
import { parseify } from 'common/util/serialize';
import { FieldParseError } from 'features/nodes/types/error';
import type { FieldInputTemplate, FieldOutputTemplate } from 'features/nodes/types/field';
import type { InvocationTemplate } from 'features/nodes/types/invocation';
import type { InvocationSchemaObject } from 'features/nodes/types/openapi';
import {
  isInvocationFieldSchema,
  isInvocationOutputSchemaObject,
  isInvocationSchemaObject,
} from 'features/nodes/types/openapi';
import { t } from 'i18next';
import { reduce } from 'lodash-es';
import type { OpenAPIV3_1 } from 'openapi-types';
import { serializeError } from 'serialize-error';

import { buildFieldInputTemplate } from './buildFieldInputTemplate';
import { buildFieldOutputTemplate } from './buildFieldOutputTemplate';
import { parseFieldType } from './parseFieldType';

const RESERVED_INPUT_FIELD_NAMES = ['id', 'type', 'use_cache'];
const RESERVED_OUTPUT_FIELD_NAMES = ['type'];
const RESERVED_FIELD_TYPES = ['IsIntermediate'];

const invocationDenylist: string[] = ['graph', 'linear_ui_output'];

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
    .filter((schema) => (nodesAllowlistExtra ? nodesAllowlistExtra.includes(schema.properties.type.default) : true))
    .filter((schema) => (nodesDenylistExtra ? !nodesDenylistExtra.includes(schema.properties.type.default) : true));

  const invocations = filteredSchemas.reduce<Record<string, InvocationTemplate>>((invocationsAccumulator, schema) => {
    const type = schema.properties.type.default;
    const title = schema.title.replace('Invocation', '');
    const tags = schema.tags ?? [];
    const description = schema.description ?? '';
    const version = schema.version;
    const nodePack = schema.node_pack;
    const classification = schema.classification;

    const inputs = reduce(
      schema.properties,
      (inputsAccumulator: Record<string, FieldInputTemplate>, property, propertyName) => {
        if (isReservedInputField(type, propertyName)) {
          logger('nodes').trace(
            { node: type, field: propertyName, schema: parseify(property) },
            'Skipped reserved input field'
          );
          return inputsAccumulator;
        }

        if (!isInvocationFieldSchema(property)) {
          logger('nodes').warn(
            { node: type, field: propertyName, schema: parseify(property) },
            'Unhandled input property'
          );
          return inputsAccumulator;
        }

        try {
          const fieldType = parseFieldType(property);

          if (isReservedFieldType(fieldType.name)) {
            logger('nodes').trace(
              { node: type, field: propertyName, schema: parseify(property) },
              'Skipped reserved input field'
            );
            return inputsAccumulator;
          }

          const fieldInputTemplate = buildFieldInputTemplate(property, propertyName, fieldType);

          inputsAccumulator[propertyName] = fieldInputTemplate;
        } catch (e) {
          if (e instanceof FieldParseError) {
            logger('nodes').warn(
              {
                node: type,
                field: propertyName,
                schema: parseify(property),
              },
              t('nodes.inputFieldTypeParseError', {
                node: type,
                field: propertyName,
                message: e.message,
              })
            );
          } else {
            logger('nodes').warn(
              {
                node: type,
                field: propertyName,
                schema: parseify(property),
                error: serializeError(e),
              },
              t('nodes.inputFieldTypeParseError', {
                node: type,
                field: propertyName,
                message: 'unknown error',
              })
            );
          }
        }

        return inputsAccumulator;
      },
      {}
    );

    const outputSchemaName = schema.output.$ref.split('/').pop();

    if (!outputSchemaName) {
      logger('nodes').warn({ outputRefObject: parseify(schema.output) }, 'No output schema name found in ref object');
      return invocationsAccumulator;
    }

    const outputSchema = openAPI.components?.schemas?.[outputSchemaName];
    if (!outputSchema) {
      logger('nodes').warn({ outputSchemaName }, 'Output schema not found');
      return invocationsAccumulator;
    }

    if (!isInvocationOutputSchemaObject(outputSchema)) {
      logger('nodes').error({ outputSchema: parseify(outputSchema) }, 'Invalid output schema');
      return invocationsAccumulator;
    }

    const outputType = outputSchema.properties.type.default;

    const outputs = reduce(
      outputSchema.properties,
      (outputsAccumulator, property, propertyName) => {
        if (!isAllowedOutputField(type, propertyName)) {
          logger('nodes').trace(
            { node: type, field: propertyName, schema: parseify(property) },
            'Skipped reserved output field'
          );
          return outputsAccumulator;
        }

        if (!isInvocationFieldSchema(property)) {
          logger('nodes').warn(
            { node: type, field: propertyName, schema: parseify(property) },
            'Unhandled output property'
          );
          return outputsAccumulator;
        }

        try {
          const fieldType = parseFieldType(property);

          if (!fieldType) {
            logger('nodes').warn(
              {
                node: type,
                field: propertyName,
                schema: parseify(property),
              },
              'Missing output field type'
            );
            return outputsAccumulator;
          }

          const fieldOutputTemplate = buildFieldOutputTemplate(property, propertyName, fieldType);

          outputsAccumulator[propertyName] = fieldOutputTemplate;
        } catch (e) {
          if (e instanceof FieldParseError) {
            logger('nodes').warn(
              {
                node: type,
                field: propertyName,
                schema: parseify(property),
              },
              t('nodes.outputFieldTypeParseError', {
                node: type,
                field: propertyName,
                message: e.message,
              })
            );
          } else {
            logger('nodes').warn(
              {
                node: type,
                field: propertyName,
                schema: parseify(property),
                error: serializeError(e),
              },
              t('nodes.outputFieldTypeParseError', {
                node: type,
                field: propertyName,
                message: 'unknown error',
              })
            );
          }
        }
        return outputsAccumulator;
      },
      {} as Record<string, FieldOutputTemplate>
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
      nodePack,
      classification,
    };

    Object.assign(invocationsAccumulator, { [type]: invocation });

    return invocationsAccumulator;
  }, {});

  return invocations;
};
