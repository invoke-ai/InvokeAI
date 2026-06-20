import type { FieldInputTemplate, FieldType } from './types';

/**
 * Field-kind helpers shared by the node editor and the Linear UI panel:
 * which field types render direct-input controls, and how handles/edges are
 * tinted by type so connections stay readable.
 */

/** Field types with a direct-input widget. Everything else is connection-only. */
const STATEFUL_FIELD_TYPE_NAMES = new Set([
  'BoardField',
  'BooleanField',
  'ColorField',
  'EnumField',
  'FloatField',
  'ImageField',
  'IntegerField',
  'ModelIdentifierField',
  'SchedulerField',
  'StringField',
]);

export const isStatefulFieldType = (type: FieldType): boolean => STATEFUL_FIELD_TYPE_NAMES.has(type.name);

const MODEL_FIELD_TYPE_NAMES = new Set([
  'CLIPField',
  'ControlLoRAField',
  'ModelIdentifierField',
  'T5EncoderField',
  'TransformerField',
  'UNetField',
  'VAEField',
]);

export const isModelFieldType = (type: FieldType): boolean => MODEL_FIELD_TYPE_NAMES.has(type.name);

/** True when the field renders an editable control on the node / linear form. */
export const isDirectInputField = (template: FieldInputTemplate): boolean =>
  template.input !== 'connection' && isStatefulFieldType(template.type) && template.type.cardinality !== 'COLLECTION';

/** A field can be exposed to the Linear UI when it can be edited directly. */
export const isExposableField = (template: FieldInputTemplate): boolean => isDirectInputField(template);

export const cloneWorkflowFieldDefault = (template: FieldInputTemplate): unknown =>
  template.default === undefined ? undefined : structuredClone(template.default);

export const isWorkflowFieldValueDefault = (template: FieldInputTemplate, value: unknown): boolean => {
  if (value === template.default) {
    return true;
  }

  if (value === undefined || template.default === undefined) {
    return false;
  }

  try {
    return JSON.stringify(value) === JSON.stringify(template.default);
  } catch {
    return false;
  }
};

const isNonEmptyString = (value: unknown): value is string => typeof value === 'string' && value.trim().length > 0;

const hasNonEmptyStringProp = (value: unknown, prop: string): boolean =>
  typeof value === 'object' && value !== null && isNonEmptyString((value as Record<string, unknown>)[prop]);

const isNumberFieldValueValid = (template: FieldInputTemplate, value: unknown): boolean => {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return false;
  }

  if (template.type.name === 'IntegerField' && !Number.isInteger(value)) {
    return false;
  }

  if (template.minimum !== null && value < template.minimum) {
    return false;
  }

  if (template.maximum !== null && value > template.maximum) {
    return false;
  }

  if (template.exclusiveMinimum !== null && value <= template.exclusiveMinimum) {
    return false;
  }

  if (template.exclusiveMaximum !== null && value >= template.exclusiveMaximum) {
    return false;
  }

  if (template.multipleOf !== null) {
    const quotient = value / template.multipleOf;

    if (Math.abs(quotient - Math.round(quotient)) > Number.EPSILON * 100) {
      return false;
    }
  }

  return true;
};

const isColorValueValid = (value: unknown): boolean => {
  if (typeof value !== 'object' || value === null) {
    return false;
  }

  const channels = value as Record<string, unknown>;

  return ['r', 'g', 'b', 'a'].every((channel) => {
    const channelValue = channels[channel];

    return typeof channelValue === 'number' && channelValue >= 0 && channelValue <= 255;
  });
};

export const isWorkflowFieldValueValid = (template: FieldInputTemplate, value: unknown): boolean => {
  switch (template.type.name) {
    case 'StringField':
      return isNonEmptyString(value);
    case 'IntegerField':
    case 'FloatField':
      return isNumberFieldValueValid(template, value);
    case 'BooleanField':
      return typeof value === 'boolean';
    case 'EnumField':
      return isNonEmptyString(value) && (template.options === null || template.options.includes(value));
    case 'ModelIdentifierField':
      return hasNonEmptyStringProp(value, 'key');
    case 'SchedulerField':
      return isNonEmptyString(value);
    case 'BoardField':
      return (
        value === undefined ||
        value === null ||
        value === 'auto' ||
        value === 'none' ||
        hasNonEmptyStringProp(value, 'board_id')
      );
    case 'ImageField':
      return hasNonEmptyStringProp(value, 'image_name');
    case 'ColorField':
      return isColorValueValid(value);
    default:
      return value !== undefined && value !== null;
  }
};

const isEmptyOptionalValue = (value: unknown): boolean => value === undefined || value === null || value === '';

export const getWorkflowFieldInvalidReason = ({
  isConnected,
  template,
  value,
}: {
  isConnected: boolean;
  template: FieldInputTemplate;
  value: unknown;
}): string | null => {
  if (isConnected) {
    return null;
  }

  if (!template.required && isEmptyOptionalValue(value)) {
    return null;
  }

  if (!template.required) {
    return isDirectInputField(template) && !isWorkflowFieldValueValid(template, value) ? 'Invalid value.' : null;
  }

  if (template.input === 'connection') {
    return 'Required connection.';
  }

  if (isWorkflowFieldValueValid(template, value)) {
    return null;
  }

  return isDirectInputField(template) ? 'Required value.' : 'Required connection.';
};

// Raw hex (not Chakra tokens) because xyflow handles are styled inline.
const FIELD_TYPE_COLORS: Record<string, string> = {
  AnyField: '#9ca3af',
  BoardField: '#a78bfa',
  BooleanField: '#4ade80',
  CLIPField: '#2dd4bf',
  ColorField: '#f472b6',
  ConditioningField: '#22d3ee',
  ControlField: '#5eead4',
  DenoiseMaskField: '#93c5fd',
  EnumField: '#60a5fa',
  FloatField: '#fb923c',
  ImageField: '#c4b5fd',
  IntegerField: '#f87171',
  LatentsField: '#f9a8d4',
  ModelIdentifierField: '#14b8a6',
  SchedulerField: '#3b82f6',
  StringField: '#facc15',
  UNetField: '#fca5a5',
  VAEField: '#2563eb',
};

const FALLBACK_COLORS = ['#06b6d4', '#a855f7', '#22c55e', '#f97316', '#ec4899', '#0d9488'];

/** Stable tint for a field type: known types get fixed colors, the rest hash into a small palette. */
export const getFieldTypeColor = (type: FieldType): string => {
  const known = FIELD_TYPE_COLORS[type.name];

  if (known) {
    return known;
  }

  let hash = 0;

  for (const char of type.name) {
    hash = (hash * 31 + char.charCodeAt(0)) % FALLBACK_COLORS.length;
  }

  return FALLBACK_COLORS[hash] as string;
};

/** Display label for a field type, e.g. `ImageField (Collection)`. */
export const getFieldTypeLabel = (type: FieldType): string => {
  const base = type.name.replace(/Field$/, '');

  if (type.cardinality === 'COLLECTION') {
    return `${base} Collection`;
  }

  if (type.cardinality === 'SINGLE_OR_COLLECTION') {
    return `${base} (single or collection)`;
  }

  return base;
};
