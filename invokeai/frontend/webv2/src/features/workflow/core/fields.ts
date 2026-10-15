import { SEED_MAX } from '@platform/core/seed';

import type { FieldInputTemplate, FieldType, WorkflowFieldInstance } from './types';

import {
  getWorkflowGeneratorInvalidReason,
  isWorkflowGeneratorFieldTypeName,
  parseWorkflowGeneratorValue,
} from './batch';

export const getEffectiveWorkflowFieldDescription = (
  instance: WorkflowFieldInstance | undefined,
  template: FieldInputTemplate | undefined
): string =>
  instance?.descriptionOverride === true
    ? (instance.description ?? '')
    : instance?.description || template?.description || '';

/** Field types with a direct-input widget. Everything else is connection-only. */
const STATEFUL_FIELD_TYPE_NAMES = new Set([
  'BoardField',
  'BooleanField',
  'ColorField',
  'EnumField',
  'FloatField',
  'FloatGeneratorField',
  'ImageField',
  'ImageGeneratorField',
  'IntegerField',
  'IntegerGeneratorField',
  // Collection loaders accept scalar or list LoRA fields; edit lists inline without extra selector/collector
  // nodes.
  'LoRAField',
  'ModelIdentifierField',
  'SchedulerField',
  'SavedWorkflowField',
  'StringField',
  'StringGeneratorField',
  'StylePresetField',
  'SystemPromptField',
  'VideoField',
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

/** Collection field types with a direct-input list widget; other collections are connection-only. */
const DIRECT_COLLECTION_FIELD_TYPE_NAMES = new Set(['FloatField', 'ImageField', 'IntegerField', 'StringField']);

/** A list the editor can author item by item. */
export const isEditableCollectionFieldType = (type: FieldType): boolean =>
  type.cardinality === 'COLLECTION' && DIRECT_COLLECTION_FIELD_TYPE_NAMES.has(type.name);

/** True when the field renders an editable control on the node / linear form. */
export const isDirectInputField = (template: FieldInputTemplate): boolean =>
  template.input !== 'connection' &&
  isStatefulFieldType(template.type) &&
  (template.type.cardinality !== 'COLLECTION' || isEditableCollectionFieldType(template.type));

/** A field can be exposed to the Linear UI when it can be edited directly. */
export const isExposableField = (template: FieldInputTemplate): boolean => isDirectInputField(template);

/** Scalar numeric fields whose linear-form element can show a randomize button. */
export const isShuffleableField = (template: FieldInputTemplate): boolean =>
  (template.type.name === 'IntegerField' || template.type.name === 'FloatField') &&
  template.type.cardinality !== 'COLLECTION' &&
  isDirectInputField(template);

const countDecimals = (value: number): number => {
  const [, fraction = ''] = String(value).split('.');

  return fraction.length;
};

const finiteNumberOrNull = (value: number | null | undefined): number | null =>
  typeof value === 'number' && Number.isFinite(value) ? value : null;

/** A random value inside the template's bounds, snapped to its step; unbounded ends default to 0…SEED_MAX. */
export const getRandomWorkflowFieldValue = (template: FieldInputTemplate, random = Math.random): number => {
  const isInteger = template.type.name === 'IntegerField';
  const multipleOf = finiteNumberOrNull(template.multipleOf);
  const step = multipleOf !== null && multipleOf > 0 ? multipleOf : isInteger ? 1 : 0;
  const exclusiveMaximum = finiteNumberOrNull(template.exclusiveMaximum);
  const exclusiveMinimum = finiteNumberOrNull(template.exclusiveMinimum);
  const maximum = finiteNumberOrNull(template.maximum);
  const minimum = finiteNumberOrNull(template.minimum);

  if (step <= 0) {
    const min = minimum ?? exclusiveMinimum ?? 0;
    const max = maximum ?? exclusiveMaximum ?? SEED_MAX;

    return min + random() * (max - min);
  }

  // Draw among the step multiples that lie inside the bounds (exclusive ends excluded), so the
  // result is always valid.
  const decimals = countDecimals(step);
  const first =
    minimum !== null
      ? Math.ceil(minimum / step)
      : exclusiveMinimum !== null
        ? Math.floor(exclusiveMinimum / step) + 1
        : 0;
  const last =
    maximum !== null
      ? Math.floor(maximum / step)
      : exclusiveMaximum !== null
        ? Math.ceil(exclusiveMaximum / step) - 1
        : Math.floor(SEED_MAX / step);
  const index = first + Math.floor(random() * (last - first + 1));

  return Number((Math.min(last, index) * step).toFixed(decimals));
};

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

/** The id inside a record-reference value such as `{ style_preset_id }`, or null when absent. */
export const getFieldRecordId = (value: unknown, prop: string): string | null =>
  hasNonEmptyStringProp(value, prop) ? ((value as Record<string, string>)[prop] as string) : null;

/** A LoRA model identifier paired with its weight — one entry of a LoRA collection field. `null` is a cleared weight. */
export interface LoraFieldCollectionEntry {
  lora: { base: string; hash: string; key: string; name: string; type: string };
  weight: number | null;
}

/**
 * The Generate LoRA weight bounds, restated: `@features/generation/settings` is side-effectful and would pull the
 * Generate settings core into the workflow boot graph. A core test pins the two together.
 */
export const LORA_FIELD_WEIGHT_RANGE = { max: 10, min: -10 } as const;

export const isLoraFieldWeightValid = (weight: unknown): weight is number =>
  typeof weight === 'number' &&
  Number.isFinite(weight) &&
  weight >= LORA_FIELD_WEIGHT_RANGE.min &&
  weight <= LORA_FIELD_WEIGHT_RANGE.max;

/** Require complete backend model identifiers; key-only entries cannot render meaningfully or enqueue successfully. */
export const isLoraFieldCollectionEntry = (value: unknown): value is LoraFieldCollectionEntry => {
  if (typeof value !== 'object' || value === null) {
    return false;
  }

  const entry = value as { lora?: unknown; weight?: unknown };

  return (
    ['base', 'hash', 'key', 'name', 'type'].every((prop) => hasNonEmptyStringProp(entry.lora, prop)) &&
    (entry.weight === null || (typeof entry.weight === 'number' && Number.isFinite(entry.weight)))
  );
};

const getLoraCollectionInvalidReason = (items: readonly unknown[]): string | null => {
  for (const [index, item] of items.entries()) {
    if (!isLoraFieldCollectionEntry(item)) {
      return `Item ${index + 1} is not a readable LoRA.`;
    }

    if (item.weight === null) {
      return `Item ${index + 1} has no weight.`;
    }

    if (!isLoraFieldWeightValid(item.weight)) {
      return `Item ${index + 1} needs a weight from ${LORA_FIELD_WEIGHT_RANGE.min} to ${LORA_FIELD_WEIGHT_RANGE.max}.`;
    }
  }

  return null;
};

/**
 * Normalize absent/scalar/list values into lists without discarding unreadable entries, which subsequent edits
 * must preserve.
 */
export const toLoraFieldCollectionList = (value: unknown): unknown[] => {
  if (Array.isArray(value)) {
    return value;
  }

  return value === undefined || value === null ? [] : [value];
};

const isNumberFieldValueValid = (template: FieldInputTemplate, value: unknown): boolean => {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return false;
  }

  if (template.type.name === 'IntegerField' && !Number.isInteger(value)) {
    return false;
  }

  const minimum = finiteNumberOrNull(template.minimum);
  const maximum = finiteNumberOrNull(template.maximum);
  const exclusiveMinimum = finiteNumberOrNull(template.exclusiveMinimum);
  const exclusiveMaximum = finiteNumberOrNull(template.exclusiveMaximum);

  if (minimum !== null && value < minimum) {
    return false;
  }

  if (maximum !== null && value > maximum) {
    return false;
  }

  if (exclusiveMinimum !== null && value <= exclusiveMinimum) {
    return false;
  }

  if (exclusiveMaximum !== null && value >= exclusiveMaximum) {
    return false;
  }

  const multipleOf = finiteNumberOrNull(template.multipleOf);

  if (multipleOf !== null && multipleOf > 0) {
    const quotient = value / multipleOf;

    if (Math.abs(quotient - Math.round(quotient)) > Number.EPSILON * 100) {
      return false;
    }
  }

  return true;
};

const isStringFieldValueValid = (template: FieldInputTemplate, value: unknown): boolean => {
  if (typeof value !== 'string') {
    return false;
  }

  const minLength = finiteNumberOrNull(template.minLength);
  const maxLength = finiteNumberOrNull(template.maxLength);

  return (minLength === null || value.length >= minLength) && (maxLength === null || value.length <= maxLength);
};

/** One entry of an editable list, judged by the template's per-item rules. */
export const isWorkflowCollectionItemValid = (template: FieldInputTemplate, item: unknown): boolean => {
  switch (template.type.name) {
    case 'StringField':
      return isStringFieldValueValid(template, item);
    case 'IntegerField':
    case 'FloatField':
      return isNumberFieldValueValid(template, item);
    case 'ImageField':
      return hasNonEmptyStringProp(item, 'image_name');
    default:
      return item !== undefined && item !== null;
  }
};

/** Count rules first, then the first bad entry, so the message names one thing to fix. */
const getCollectionInvalidReason = (template: FieldInputTemplate, items: readonly unknown[]): string | null => {
  const minItems = finiteNumberOrNull(template.minItems);
  const maxItems = finiteNumberOrNull(template.maxItems);

  if (minItems !== null && minItems > 0 && items.length === 0) {
    return 'Collection is empty.';
  }

  if (minItems !== null && items.length < minItems) {
    return `Needs at least ${minItems} items.`;
  }

  if (maxItems !== null && items.length > maxItems) {
    return `Allows at most ${maxItems} items.`;
  }

  const badIndex = items.findIndex((item) => !isWorkflowCollectionItemValid(template, item));

  if (badIndex === -1) {
    return null;
  }

  return items[badIndex] === null || items[badIndex] === undefined
    ? `Item ${badIndex + 1} is empty.`
    : `Item ${badIndex + 1} is invalid.`;
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
  if (isWorkflowGeneratorFieldTypeName(template.type.name)) {
    const generator = parseWorkflowGeneratorValue(template.type.name, value);

    return generator !== null && getWorkflowGeneratorInvalidReason(generator) === null;
  }

  if (
    template.type.cardinality === 'COLLECTION' &&
    (template.type.name === 'StringField' ||
      template.type.name === 'IntegerField' ||
      template.type.name === 'FloatField' ||
      template.type.name === 'ImageField')
  ) {
    return Array.isArray(value) && getCollectionInvalidReason(template, value) === null;
  }

  switch (template.type.name) {
    case 'SavedWorkflowField':
      return typeof value === 'string';
    case 'StringField':
      // An empty string is a legitimate string value (e.g. a blank negative prompt).
      return isStringFieldValueValid(template, value);
    case 'IntegerField':
    case 'FloatField':
      return isNumberFieldValueValid(template, value);
    case 'BooleanField':
      return typeof value === 'boolean';
    case 'EnumField':
      if (value === undefined || value === null) {
        return !template.required;
      }

      return (
        (isNonEmptyString(value) ||
          (typeof value === 'number' && Number.isFinite(value)) ||
          typeof value === 'boolean') &&
        (template.options === null || template.options.includes(value))
      );
    case 'ModelIdentifierField':
      return hasNonEmptyStringProp(value, 'key');
    case 'LoRAField':
      // An empty list is a legitimate value: a collection loader with no LoRAs passes its
      // models through untouched.
      return (
        (Array.isArray(value) || isLoraFieldCollectionEntry(value)) &&
        getLoraCollectionInvalidReason(toLoraFieldCollectionList(value)) === null
      );
    case 'SchedulerField':
      return isNonEmptyString(value);
    case 'StylePresetField':
      return hasNonEmptyStringProp(value, 'style_preset_id');
    case 'SystemPromptField':
      return hasNonEmptyStringProp(value, 'system_prompt_id');
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
    case 'VideoField':
      // COLLECTION video values are arrays no direct-input widget authors; keep the
      // generic non-null check so persisted values (imported workflows) stay valid.
      if (template.type.cardinality === 'COLLECTION') {
        return value !== undefined && value !== null;
      }

      return hasNonEmptyStringProp(value, 'video_name');
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

  // An editable list reports which count or entry rule it breaks; the generic reasons cannot say.
  if (template.type.cardinality === 'COLLECTION' && Array.isArray(value) && isDirectInputField(template)) {
    return getCollectionInvalidReason(template, value);
  }

  if (template.type.name === 'LoRAField' && isDirectInputField(template) && !isEmptyOptionalValue(value)) {
    return getLoraCollectionInvalidReason(toLoraFieldCollectionList(value));
  }

  if (
    isWorkflowGeneratorFieldTypeName(template.type.name) &&
    isDirectInputField(template) &&
    !isEmptyOptionalValue(value)
  ) {
    const generator = parseWorkflowGeneratorValue(template.type.name, value);

    return generator === null ? 'Invalid value.' : getWorkflowGeneratorInvalidReason(generator);
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

  if (!isDirectInputField(template)) {
    return 'Required connection.';
  }

  return isEmptyOptionalValue(value) ? 'Required value.' : 'Invalid value.';
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
  LoRAField: '#e879f9',
  ModelIdentifierField: '#14b8a6',
  SchedulerField: '#3b82f6',
  SavedWorkflowField: '#818cf8',
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
