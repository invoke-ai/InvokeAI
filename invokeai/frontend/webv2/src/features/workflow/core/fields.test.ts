import { describe, expect, it } from 'vitest';

import type { FieldInputTemplate, FieldType } from './types';

import {
  getWorkflowFieldInvalidReason,
  getWorkflowFieldIssue,
  isDirectInputField,
  isLoraFieldCollectionEntry,
  isWorkflowCollectionItemValid,
  isWorkflowFieldValueValid,
  toLoraFieldCollectionList,
  getRandomWorkflowFieldValue,
  isShuffleableField,
} from './fields';

const single = (name: string): FieldType => ({ batch: false, cardinality: 'SINGLE', name });

const LORA_ENTRY = {
  lora: { base: 'sd-1', hash: 'hash', key: 'lora-key', name: 'LoRA', type: 'lora' },
  weight: 0.75,
};

const input = (overrides: Partial<FieldInputTemplate> = {}): FieldInputTemplate => ({
  default: undefined,
  description: '',
  exclusiveMaximum: null,
  exclusiveMinimum: null,
  fieldKind: 'input',
  input: 'any',
  maximum: null,
  minimum: null,
  multipleOf: null,
  name: 'value',
  options: null,
  required: true,
  title: 'Value',
  type: single('StringField'),
  uiChoiceLabels: null,
  uiComponent: null,
  uiHidden: false,
  uiModelBase: null,
  uiModelFormat: null,
  uiModelType: null,
  uiOrder: null,
  ...overrides,
});

describe('workflow field validation', () => {
  it('flags missing required direct values and ignores optional fields', () => {
    expect(getWorkflowFieldInvalidReason({ isConnected: false, template: input(), value: undefined })).toBe(
      'Required value.'
    );
    expect(getWorkflowFieldInvalidReason({ isConnected: false, template: input({ required: false }), value: '' })).toBe(
      null
    );
  });

  it('flags missing required connections', () => {
    expect(
      getWorkflowFieldInvalidReason({ isConnected: false, template: input({ input: 'connection' }), value: undefined })
    ).toBe('Required connection.');
    expect(
      getWorkflowFieldInvalidReason({ isConnected: false, template: input({ input: 'connection' }), value: 'value' })
    ).toBe('Required connection.');
  });

  it('names what a blocking field lacks, so callers can say what to do about it', () => {
    const integer = input({ minimum: 1, type: single('IntegerField') });

    expect(getWorkflowFieldIssue({ isConnected: false, template: integer, value: undefined })?.kind).toBe(
      'missingValue'
    );
    expect(getWorkflowFieldIssue({ isConnected: false, template: integer, value: 0 })).toEqual({
      kind: 'invalidValue',
      reason: 'Invalid value.',
    });
    expect(
      getWorkflowFieldIssue({ isConnected: false, template: input({ input: 'connection' }), value: undefined })?.kind
    ).toBe('missingConnection');
    expect(getWorkflowFieldIssue({ isConnected: false, template: integer, value: 3 })).toBeNull();
  });

  it('validates numeric constraints', () => {
    const template = input({ maximum: 10, minimum: 1, type: single('IntegerField') });

    expect(isWorkflowFieldValueValid(template, 5)).toBe(true);
    expect(isWorkflowFieldValueValid(template, 0)).toBe(false);
    expect(isWorkflowFieldValueValid(template, 5.5)).toBe(false);
  });

  it('validates object-backed model and image fields', () => {
    expect(isWorkflowFieldValueValid(input({ type: single('ModelIdentifierField') }), { key: 'model-key' })).toBe(true);
    expect(isWorkflowFieldValueValid(input({ type: single('ModelIdentifierField') }), {})).toBe(false);
    expect(isWorkflowFieldValueValid(input({ type: single('ImageField') }), { image_name: 'image.png' })).toBe(true);
    expect(isWorkflowFieldValueValid(input({ type: single('ImageField') }), { image_name: '' })).toBe(false);
  });

  describe('scalar collections', () => {
    const list = (name: string, overrides: Partial<FieldInputTemplate> = {}) =>
      input({ type: { batch: false, cardinality: 'COLLECTION', name }, ...overrides });
    const reason = (template: FieldInputTemplate, value: unknown) =>
      getWorkflowFieldInvalidReason({ isConnected: false, template, value });

    it('validates each entry with the scalar rules and reports the first bad one by position', () => {
      const integers = list('IntegerField', { minimum: 0, multipleOf: 2 });

      expect(isWorkflowFieldValueValid(integers, [0, 2, 4])).toBe(true);
      expect(isWorkflowFieldValueValid(integers, [])).toBe(true);
      expect(isWorkflowFieldValueValid(integers, 2)).toBe(false);
      expect(reason(integers, [2, 3])).toBe('Item 2 is invalid.');
      expect(reason(integers, [2, -2])).toBe('Item 2 is invalid.');
      expect(reason(integers, [2, 2.5])).toBe('Item 2 is invalid.');
      expect(reason(integers, [null, 2])).toBe('Item 1 is empty.');
      expect(isWorkflowCollectionItemValid(integers, 3)).toBe(false);
      expect(isWorkflowCollectionItemValid(integers, 4)).toBe(true);

      const strings = list('StringField', { maxLength: 3, minLength: 1 });

      expect(isWorkflowFieldValueValid(strings, ['a', 'abc'])).toBe(true);
      expect(reason(strings, ['a', ''])).toBe('Item 2 is invalid.');
      expect(reason(strings, ['abcd'])).toBe('Item 1 is invalid.');
      expect(reason(strings, ['a', 7])).toBe('Item 2 is invalid.');
    });
  });

  it('reports a generator setting that cannot run through the field, as the widget shows it', () => {
    const generator = input({
      type: { batch: false, cardinality: 'SINGLE', name: 'FloatGeneratorField' },
    });
    const reasonFor = (value: unknown) =>
      getWorkflowFieldInvalidReason({ isConnected: false, template: generator, value });

    expect(reasonFor({ count: 3, start: 0, step: 0.5, type: 'float_generator_arithmetic_sequence' })).toBeNull();
    expect(reasonFor({ count: null, start: 0, step: 0.5, type: 'float_generator_arithmetic_sequence' })).toBe(
      'Count is empty.'
    );
    expect(reasonFor({ count: 3, start: 'x', type: 'float_generator_arithmetic_sequence' })).toBe('Invalid value.');
    expect(
      isWorkflowFieldValueValid(generator, {
        count: 0,
        start: 0,
        step: 0.5,
        type: 'float_generator_arithmetic_sequence',
      })
    ).toBe(false);
  });

  it('rejects a LoRA identifier missing the fields the backend requires, rather than enqueuing a 422', () => {
    const loras = input({
      required: false,
      type: { batch: false, cardinality: 'SINGLE_OR_COLLECTION', name: 'LoRAField' },
    });

    // A key alone would render a nameless row and be rejected at enqueue time.
    expect(isWorkflowFieldValueValid(loras, [{ lora: { key: 'lora-key' }, weight: 0.75 }])).toBe(false);
    expect(isWorkflowFieldValueValid(loras, [{ ...LORA_ENTRY, lora: { ...LORA_ENTRY.lora, hash: '' } }])).toBe(false);
    expect(getWorkflowFieldInvalidReason({ isConnected: false, template: loras, value: [{ weight: 1 }] })).toBe(
      'Item 1 is not a readable LoRA.'
    );
  });

  it('treats a generator field as editable and judges its value by the generator schema', () => {
    const generator = input({ type: single('FloatGeneratorField') });

    expect(isDirectInputField(generator)).toBe(true);
    expect(
      isWorkflowFieldValueValid(generator, { count: 2, start: 0, step: 1, type: 'float_generator_arithmetic_sequence' })
    ).toBe(true);
    expect(isWorkflowFieldValueValid(generator, { count: 0, type: 'float_generator_arithmetic_sequence' })).toBe(false);
    expect(isWorkflowFieldValueValid(generator, undefined)).toBe(false);
    expect(
      getWorkflowFieldInvalidReason({
        isConnected: false,
        template: generator,
        value: { type: 'integer_generator_parse_string' },
      })
    ).toBe('Invalid value.');
  });
});

describe('LoRA collection values', () => {
  it('preserves unreadable items verbatim so an edit cannot silently delete them', () => {
    // Keep unreadable collection entries so editing another row cannot destroy hand-authored data.
    const items = [LORA_ENTRY, { lora: { key: 'ghost' }, weight: 1 }, null, 'lora'];

    expect(toLoraFieldCollectionList(items)).toEqual(items);
    expect(items.filter(isLoraFieldCollectionEntry)).toEqual([LORA_ENTRY]);
  });
});

describe('getRandomWorkflowFieldValue', () => {
  it('stays inside the template bounds and snaps to the step', () => {
    const template = input({ maximum: 10, minimum: 2, multipleOf: 2, type: single('IntegerField') });

    expect(getRandomWorkflowFieldValue(template, () => 0)).toBe(2);
    expect(getRandomWorkflowFieldValue(template, () => 0.999)).toBe(10);
    expect(getRandomWorkflowFieldValue(template, () => 0.55)).toBe(6);

    const odd = input({ maximum: 9, minimum: 0, multipleOf: 2, type: single('IntegerField') });

    expect(getRandomWorkflowFieldValue(odd, () => 0.999)).toBe(8);

    const decimal = input({ maximum: 1, minimum: 0, multipleOf: 0.1, type: single('FloatField') });

    expect(getRandomWorkflowFieldValue(decimal, () => 0.3)).toBe(0.3);

    const tiny = input({ maximum: 9e-7, minimum: 1e-7, multipleOf: 1e-7, type: single('FloatField') });

    expect(getRandomWorkflowFieldValue(tiny, () => 0.5)).toBe(5e-7);
  });

  it('shuffles only direct numeric fields with a value in range', () => {
    expect(isShuffleableField(input({ type: single('IntegerField') }))).toBe(true);
    expect(isShuffleableField(input({ type: single('FloatField') }))).toBe(true);
    expect(isShuffleableField(input({ input: 'connection', type: single('IntegerField') }))).toBe(false);
    expect(isShuffleableField(input())).toBe(false);
    // No multiple of 8 lies between 1 and 7, so any value a shuffle wrote would be invalid.
    expect(isShuffleableField(input({ maximum: 7, minimum: 1, multipleOf: 8, type: single('IntegerField') }))).toBe(
      false
    );
  });
});
