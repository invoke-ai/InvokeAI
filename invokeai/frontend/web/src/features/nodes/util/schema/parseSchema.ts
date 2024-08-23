import { logger } from 'app/logging/logger';
import type { SerializableObject } from 'common/types';
import { deepClone } from 'common/util/deepClone';
import { parseify } from 'common/util/serialize';
import type { Templates } from 'features/nodes/store/types';
import { FieldParseError } from 'features/nodes/types/error';
import {
  type FieldInputTemplate,
  type FieldOutputTemplate,
  type FieldType,
  isStatefulFieldType,
} from 'features/nodes/types/field';
import type { InvocationTemplate } from 'features/nodes/types/invocation';
import type { InvocationFieldSchema, InvocationSchemaObject } from 'features/nodes/types/openapi';
import {
  isInvocationFieldSchema,
  isInvocationOutputSchemaObject,
  isInvocationSchemaObject,
} from 'features/nodes/types/openapi';
import { t } from 'i18next';
import { isEqual, reduce } from 'lodash-es';
import type { OpenAPIV3_1 } from 'openapi-types';
import { serializeError } from 'serialize-error';

import { buildFieldInputTemplate } from './buildFieldInputTemplate';
import { buildFieldOutputTemplate } from './buildFieldOutputTemplate';
import { isCollectionFieldType, parseFieldType } from './parseFieldType';

const log = logger('system');

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
): Templates => {
  const filteredSchemas = Object.values(openAPI.components?.schemas ?? {})
    .filter(isInvocationSchemaObject)
    .filter(isNotInDenylist)
    .filter((schema) => (nodesAllowlistExtra ? nodesAllowlistExtra.includes(schema.properties.type.default) : true))
    .filter((schema) => (nodesDenylistExtra ? !nodesDenylistExtra.includes(schema.properties.type.default) : true));

  const invocations = filteredSchemas.reduce<Templates>((invocationsAccumulator, schema) => {
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
          log.trace(
            { node: type, field: propertyName, schema: property } as SerializableObject,
            'Skipped reserved input field'
          );
          return inputsAccumulator;
        }

        if (!isInvocationFieldSchema(property)) {
          log.warn({ node: type, field: propertyName, schema: parseify(property) }, 'Unhandled input property');
          return inputsAccumulator;
        }

        const fieldTypeOverride: FieldType | null = property.ui_type
          ? {
              name: property.ui_type,
              cardinality: isCollectionFieldType(property.ui_type) ? 'COLLECTION' : 'SINGLE',
            }
          : null;

        const originalFieldType = getFieldType(property, propertyName, type, 'input');

        const fieldType = fieldTypeOverride ?? originalFieldType;
        if (!fieldType) {
          log.trace({ node: type, field: propertyName, schema: parseify(property) }, 'Unable to parse field type');
          return inputsAccumulator;
        }

        if (isReservedFieldType(fieldType.name)) {
          log.trace({ node: type, field: propertyName, schema: parseify(property) }, 'Skipped reserved input field');
          return inputsAccumulator;
        }

        if (isStatefulFieldType(fieldType) && originalFieldType && !isEqual(originalFieldType, fieldType)) {
          fieldType.originalType = deepClone(originalFieldType);
        }

        const fieldInputTemplate = buildFieldInputTemplate(property, propertyName, fieldType);
        inputsAccumulator[propertyName] = fieldInputTemplate;

        return inputsAccumulator;
      },
      {}
    );

    const outputSchemaName = schema.output.$ref.split('/').pop();

    if (!outputSchemaName) {
      log.warn({ outputRefObject: parseify(schema.output) }, 'No output schema name found in ref object');
      return invocationsAccumulator;
    }

    const outputSchema = openAPI.components?.schemas?.[outputSchemaName];
    if (!outputSchema) {
      log.warn({ outputSchemaName }, 'Output schema not found');
      return invocationsAccumulator;
    }

    if (!isInvocationOutputSchemaObject(outputSchema)) {
      log.error({ outputSchema: parseify(outputSchema) }, 'Invalid output schema');
      return invocationsAccumulator;
    }

    const outputType = outputSchema.properties.type.default;

    const outputs = reduce(
      outputSchema.properties,
      (outputsAccumulator, property, propertyName) => {
        if (!isAllowedOutputField(type, propertyName)) {
          log.trace({ node: type, field: propertyName, schema: parseify(property) }, 'Skipped reserved output field');
          return outputsAccumulator;
        }

        if (!isInvocationFieldSchema(property)) {
          log.warn({ node: type, field: propertyName, schema: parseify(property) }, 'Unhandled output property');
          return outputsAccumulator;
        }

        const fieldTypeOverride: FieldType | null = property.ui_type
          ? {
              name: property.ui_type,
              cardinality: isCollectionFieldType(property.ui_type) ? 'COLLECTION' : 'SINGLE',
            }
          : null;

        const originalFieldType = getFieldType(property, propertyName, type, 'output');

        const fieldType = fieldTypeOverride ?? originalFieldType;
        if (!fieldType) {
          log.trace({ node: type, field: propertyName, schema: parseify(property) }, 'Unable to parse field type');
          return outputsAccumulator;
        }

        if (isStatefulFieldType(fieldType) && originalFieldType && !isEqual(originalFieldType, fieldType)) {
          fieldType.originalType = deepClone(originalFieldType);
        }

        const fieldOutputTemplate = buildFieldOutputTemplate(property, propertyName, fieldType);

        outputsAccumulator[propertyName] = fieldOutputTemplate;
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

const getFieldType = (
  property: InvocationFieldSchema,
  propertyName: string,
  type: string,
  kind: 'input' | 'output'
): FieldType | null => {
  try {
    return parseFieldType(property);
  } catch (e) {
    const tKey = kind === 'input' ? 'nodes.inputFieldTypeParseError' : 'nodes.outputFieldTypeParseError';
    if (e instanceof FieldParseError) {
      log.warn(
        {
          node: type,
          field: propertyName,
          schema: parseify(property),
        },
        t(tKey, {
          node: type,
          field: propertyName,
          message: e.message,
        })
      );
    } else {
      log.warn(
        {
          node: type,
          field: propertyName,
          schema: parseify(property),
          error: serializeError(e),
        },
        t(tKey, {
          node: type,
          field: propertyName,
          message: 'unknown error',
        })
      );
    }
    return null;
  }
};
