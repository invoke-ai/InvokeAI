import type { NodesState, Templates } from 'features/nodes/store/types';
import type {
  FieldInputInstance,
  FieldInputTemplate,
  FloatFieldCollectionInputTemplate,
  FloatFieldCollectionValue,
  ImageFieldCollectionInputTemplate,
  ImageFieldCollectionValue,
  IntegerFieldCollectionInputTemplate,
  IntegerFieldCollectionValue,
  StringFieldCollectionInputTemplate,
  StringFieldCollectionValue,
} from 'features/nodes/types/field';
import {
  isFloatFieldCollectionInputInstance,
  isFloatFieldCollectionInputTemplate,
  isImageFieldCollectionInputInstance,
  isImageFieldCollectionInputTemplate,
  isIntegerFieldCollectionInputInstance,
  isIntegerFieldCollectionInputTemplate,
  isStringFieldCollectionInputInstance,
  isStringFieldCollectionInputTemplate,
} from 'features/nodes/types/field';
import { type InvocationNode, type InvocationTemplate, isInvocationNode } from 'features/nodes/types/invocation';
import { t } from 'i18next';
import { assert } from 'tsafe';

const validateImageFieldCollectionValue = (
  value: NonNullable<ImageFieldCollectionValue>,
  template: ImageFieldCollectionInputTemplate
): string[] => {
  const reasons: string[] = [];
  const { minItems, maxItems } = template;
  const count = value.length;

  // Image collections may have min or max items to validate
  if (minItems !== undefined && minItems > 0 && count === 0) {
    reasons.push(t('parameters.invoke.collectionEmpty'));
  }

  if (minItems !== undefined && count < minItems) {
    reasons.push(t('parameters.invoke.collectionTooFewItems', { count, minItems }));
  }

  if (maxItems !== undefined && count > maxItems) {
    reasons.push(t('parameters.invoke.collectionTooManyItems', { count, maxItems }));
  }

  return reasons;
};

const validateStringFieldCollectionValue = (
  value: NonNullable<StringFieldCollectionValue>,
  template: StringFieldCollectionInputTemplate
): string[] => {
  const reasons: string[] = [];
  const { minItems, maxItems, minLength, maxLength } = template;
  const count = value.length;

  // Image collections may have min or max items to validate
  if (minItems !== undefined && minItems > 0 && count === 0) {
    reasons.push(t('parameters.invoke.collectionEmpty'));
  }

  if (minItems !== undefined && count < minItems) {
    reasons.push(t('parameters.invoke.collectionTooFewItems', { count, minItems }));
  }

  if (maxItems !== undefined && count > maxItems) {
    reasons.push(t('parameters.invoke.collectionTooManyItems', { count, maxItems }));
  }

  for (const str of value) {
    if (maxLength !== undefined && str.length > maxLength) {
      reasons.push(t('parameters.invoke.collectionStringTooLong', { value, maxLength }));
    }
    if (minLength !== undefined && str.length < minLength) {
      reasons.push(t('parameters.invoke.collectionStringTooShort', { value, minLength }));
    }
  }

  return reasons;
};

const validateNumberFieldCollectionValue = (
  value: NonNullable<IntegerFieldCollectionValue> | NonNullable<FloatFieldCollectionValue>,
  template: IntegerFieldCollectionInputTemplate | FloatFieldCollectionInputTemplate
): string[] => {
  const reasons: string[] = [];
  const { minItems, maxItems, minimum, maximum, exclusiveMinimum, exclusiveMaximum, multipleOf } = template;
  const count = value.length;

  // Image collections may have min or max items to validate
  if (minItems !== undefined && minItems > 0 && count === 0) {
    reasons.push(t('parameters.invoke.collectionEmpty'));
  }

  if (minItems !== undefined && count < minItems) {
    reasons.push(t('parameters.invoke.collectionTooFewItems', { count, minItems }));
  }

  if (maxItems !== undefined && count > maxItems) {
    reasons.push(t('parameters.invoke.collectionTooManyItems', { count, maxItems }));
  }

  for (const num of value) {
    if (maximum !== undefined && num > maximum) {
      reasons.push(t('parameters.invoke.collectionNumberGTMax', { value, maximum }));
    }
    if (minimum !== undefined && num < minimum) {
      reasons.push(t('parameters.invoke.collectionNumberLTMin', { value, minimum }));
    }
    if (exclusiveMaximum !== undefined && num >= exclusiveMaximum) {
      reasons.push(t('parameters.invoke.collectionNumberGTExclusiveMax', { value, exclusiveMaximum }));
    }
    if (exclusiveMinimum !== undefined && num <= exclusiveMinimum) {
      reasons.push(t('parameters.invoke.collectionNumberLTExclusiveMin', { value, exclusiveMinimum }));
    }
    if (multipleOf !== undefined && num % multipleOf !== 0) {
      reasons.push(t('parameters.invoke.collectionNumberNotMultipleOf', { value, multipleOf }));
    }
  }

  return reasons;
};

type NodeError = {
  type: 'node-error';
  nodeId: string;
  issue: string;
};

type FieldError = {
  type: 'field-error';
  nodeId: string;
  fieldName: string;
  prefix: string;
  issue: string;
};

const getFieldErrorPrefix = (
  node: InvocationNode,
  nodeTemplate: InvocationTemplate,
  field: FieldInputInstance,
  fieldTemplate: FieldInputTemplate
): string => {
  return `${node.data.label || nodeTemplate.title} -> ${field.label || fieldTemplate.title}`;
};

export const getFieldErrors = (
  node: InvocationNode,
  nodeTemplate: InvocationTemplate,
  field: FieldInputInstance,
  fieldTemplate: FieldInputTemplate,
  nodesState: NodesState
): FieldError[] => {
  const errors: FieldError[] = [];
  const prefix = getFieldErrorPrefix(node, nodeTemplate, field, fieldTemplate);

  const nodeId = node.data.id;
  const fieldName = field.name;

  const isConnected =
    nodesState.edges.find((edge) => {
      return edge.target === nodeId && edge.targetHandle === fieldName;
    }) !== undefined;

  // 'connection' input fields have no data validation - only connection validation
  if (fieldTemplate.required && !isConnected && field.value === undefined) {
    errors.push({
      type: 'field-error',
      nodeId,
      fieldName,
      prefix,
      issue: t('parameters.invoke.missingInputForField'),
    });
  } else if (
    field.value &&
    isImageFieldCollectionInputTemplate(fieldTemplate) &&
    isImageFieldCollectionInputInstance(field)
  ) {
    const issues = validateImageFieldCollectionValue(field.value, fieldTemplate);
    errors.push(
      ...issues.map<FieldError>((issue) => ({
        type: 'field-error',
        nodeId,
        fieldName,
        prefix,
        issue,
      }))
    );
  } else if (
    field.value &&
    isStringFieldCollectionInputTemplate(fieldTemplate) &&
    isStringFieldCollectionInputInstance(field)
  ) {
    const issues = validateStringFieldCollectionValue(field.value, fieldTemplate);
    errors.push(
      ...issues.map<FieldError>((issue) => ({
        type: 'field-error',
        nodeId,
        fieldName,
        prefix,
        issue,
      }))
    );
  } else if (
    field.value &&
    isIntegerFieldCollectionInputTemplate(fieldTemplate) &&
    isIntegerFieldCollectionInputInstance(field)
  ) {
    const issues = validateNumberFieldCollectionValue(field.value, fieldTemplate);
    errors.push(
      ...issues.map<FieldError>((issue) => ({
        type: 'field-error',
        nodeId,
        fieldName,
        prefix,
        issue,
      }))
    );
  } else if (
    field.value &&
    isFloatFieldCollectionInputTemplate(fieldTemplate) &&
    isFloatFieldCollectionInputInstance(field)
  ) {
    const issues = validateNumberFieldCollectionValue(field.value, fieldTemplate);
    errors.push(
      ...issues.map<FieldError>((issue) => ({
        type: 'field-error',
        nodeId,
        fieldName,
        prefix,
        issue,
      }))
    );
  }

  return errors;
};

export const getInvocationNodeErrors = (
  nodeId: string,
  templates: Templates,
  nodesState: NodesState
): (NodeError | FieldError)[] => {
  const errors: (NodeError | FieldError)[] = [];

  const node = nodesState.nodes.find((node) => node.id === nodeId);

  assert(isInvocationNode(node), `Node ${nodeId} is not an invocation node`);

  const nodeTemplate = templates[node.data.type];

  if (!nodeTemplate) {
    errors.push({ type: 'node-error', nodeId, issue: t('parameters.invoke.missingNodeTemplate') });
    return errors;
  }

  for (const [fieldName, field] of Object.entries(node.data.inputs)) {
    const fieldTemplate = nodeTemplate.inputs[fieldName];

    if (!fieldTemplate) {
      errors.push({ type: 'node-error', nodeId, issue: t('parameters.invoke.missingFieldTemplate') });
      continue;
    }

    errors.push(...getFieldErrors(node, nodeTemplate, field, fieldTemplate, nodesState));
  }

  return errors;
};
