import type { NodesState, Templates } from 'features/nodes/store/types';
import type {
  FieldInputInstance,
  FieldInputTemplate,
  FloatFieldCollectionInputTemplate,
  FloatFieldCollectionValue,
  FloatFieldInputTemplate,
  FloatFieldValue,
  ImageFieldCollectionInputTemplate,
  ImageFieldCollectionValue,
  IntegerFieldCollectionInputTemplate,
  IntegerFieldCollectionValue,
  IntegerFieldInputTemplate,
  IntegerFieldValue,
  StatefulFieldValue,
  StringFieldCollectionInputTemplate,
  StringFieldCollectionValue,
} from 'features/nodes/types/field';
import {
  isFloatFieldCollectionInputInstance,
  isFloatFieldCollectionInputTemplate,
  isFloatFieldInputInstance,
  isFloatFieldInputTemplate,
  isImageFieldCollectionInputInstance,
  isImageFieldCollectionInputTemplate,
  isIntegerFieldCollectionInputInstance,
  isIntegerFieldCollectionInputTemplate,
  isIntegerFieldInputInstance,
  isIntegerFieldInputTemplate,
  isStringFieldCollectionInputInstance,
  isStringFieldCollectionInputTemplate,
} from 'features/nodes/types/field';
import { type InvocationNode, type InvocationTemplate, isInvocationNode } from 'features/nodes/types/invocation';
import { t } from 'i18next';
import { assert } from 'tsafe';

type FieldValidationFunc<TValue extends StatefulFieldValue, TTemplate extends FieldInputTemplate> = (
  value: TValue,
  template: TTemplate
) => string[];

const validateImageFieldCollectionValue: FieldValidationFunc<
  NonNullable<ImageFieldCollectionValue>,
  ImageFieldCollectionInputTemplate
> = (value, template) => {
  const reasons: string[] = [];
  const { minItems, maxItems } = template;
  const count = value.length;

  // Image collections may have min or max items to validate
  if (minItems !== undefined && minItems > 0 && count === 0) {
    reasons.push(t('parameters.invoke.collectionEmpty'));
  } else {
    if (minItems !== undefined && count < minItems) {
      reasons.push(t('parameters.invoke.collectionTooFewItems', { count, minItems }));
    }

    if (maxItems !== undefined && count > maxItems) {
      reasons.push(t('parameters.invoke.collectionTooManyItems', { count, maxItems }));
    }
  }

  return reasons;
};

const validateStringFieldCollectionValue: FieldValidationFunc<
  NonNullable<StringFieldCollectionValue>,
  StringFieldCollectionInputTemplate
> = (value, template) => {
  const reasons: string[] = [];
  const { minItems, maxItems, minLength, maxLength } = template;
  const count = value.length;

  // Image collections may have min or max items to validate
  if (minItems !== undefined && minItems > 0 && count === 0) {
    reasons.push(t('parameters.invoke.collectionEmpty'));
  } else {
    if (minItems !== undefined && count < minItems) {
      reasons.push(t('parameters.invoke.collectionTooFewItems', { count, minItems }));
    }

    if (maxItems !== undefined && count > maxItems) {
      reasons.push(t('parameters.invoke.collectionTooManyItems', { count, maxItems }));
    }
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

const validateNumberFieldCollectionValue: FieldValidationFunc<
  NonNullable<IntegerFieldCollectionValue> | NonNullable<FloatFieldCollectionValue>,
  IntegerFieldCollectionInputTemplate | FloatFieldCollectionInputTemplate
> = (value, template) => {
  const reasons: string[] = [];
  const { minItems, maxItems, minimum, maximum, exclusiveMinimum, exclusiveMaximum, multipleOf } = template;
  const count = value.length;

  // Image collections may have min or max items to validate
  if (minItems !== undefined && minItems > 0 && count === 0) {
    reasons.push(t('parameters.invoke.collectionEmpty'));
  } else {
    if (minItems !== undefined && count < minItems) {
      reasons.push(t('parameters.invoke.collectionTooFewItems', { count, minItems }));
    }

    if (maxItems !== undefined && count > maxItems) {
      reasons.push(t('parameters.invoke.collectionTooManyItems', { count, maxItems }));
    }
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

const validateNumberFieldValue: FieldValidationFunc<
  FloatFieldValue | IntegerFieldValue,
  FloatFieldInputTemplate | IntegerFieldInputTemplate
> = (value, template) => {
  const reasons: string[] = [];
  const { minimum, maximum, exclusiveMinimum, exclusiveMaximum, multipleOf } = template;

  if (maximum !== undefined && value > maximum) {
    reasons.push(t('parameters.invoke.collectionNumberGTMax', { value, maximum }));
  }
  if (minimum !== undefined && value < minimum) {
    reasons.push(t('parameters.invoke.collectionNumberLTMin', { value, minimum }));
  }
  if (exclusiveMaximum !== undefined && value >= exclusiveMaximum) {
    reasons.push(t('parameters.invoke.collectionNumberGTExclusiveMax', { value, exclusiveMaximum }));
  }
  if (exclusiveMinimum !== undefined && value <= exclusiveMinimum) {
    reasons.push(t('parameters.invoke.collectionNumberLTExclusiveMin', { value, exclusiveMinimum }));
  }
  if (multipleOf !== undefined && value % multipleOf !== 0) {
    reasons.push(t('parameters.invoke.collectionNumberNotMultipleOf', { value, multipleOf }));
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

const getIssuesToFieldErrorsMapFunc =
  (nodeId: string, fieldName: string, prefix: string): ((issue: string) => FieldError) =>
  (issue: string) => ({
    type: 'field-error',
    nodeId,
    fieldName,
    prefix,
    issue,
  });

export const getFieldErrors = (
  node: InvocationNode,
  nodeTemplate: InvocationTemplate,
  field: FieldInputInstance,
  fieldTemplate: FieldInputTemplate,
  nodesState: NodesState
): FieldError[] => {
  const errors: FieldError[] = [];
  const prefix = getFieldErrorPrefix(node, nodeTemplate, field, fieldTemplate);
  const issueToFieldError = getIssuesToFieldErrorsMapFunc(node.data.id, field.name, prefix);

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
  } else if (isConnected) {
    // Connected fields have no value to validate - they are OK
  } else if (field.value !== undefined) {
    if (isImageFieldCollectionInputTemplate(fieldTemplate) && isImageFieldCollectionInputInstance(field)) {
      errors.push(...validateImageFieldCollectionValue(field.value, fieldTemplate).map(issueToFieldError));
    } else if (isStringFieldCollectionInputTemplate(fieldTemplate) && isStringFieldCollectionInputInstance(field)) {
      errors.push(...validateStringFieldCollectionValue(field.value, fieldTemplate).map(issueToFieldError));
    } else if (isIntegerFieldCollectionInputTemplate(fieldTemplate) && isIntegerFieldCollectionInputInstance(field)) {
      errors.push(...validateNumberFieldCollectionValue(field.value, fieldTemplate).map(issueToFieldError));
    } else if (isFloatFieldCollectionInputTemplate(fieldTemplate) && isFloatFieldCollectionInputInstance(field)) {
      errors.push(...validateNumberFieldCollectionValue(field.value, fieldTemplate).map(issueToFieldError));
    } else if (isFloatFieldInputTemplate(fieldTemplate) && isFloatFieldInputInstance(field)) {
      errors.push(...validateNumberFieldValue(field.value, fieldTemplate).map(issueToFieldError));
    } else if (isIntegerFieldInputTemplate(fieldTemplate) && isIntegerFieldInputInstance(field)) {
      errors.push(...validateNumberFieldValue(field.value, fieldTemplate).map(issueToFieldError));
    }
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
