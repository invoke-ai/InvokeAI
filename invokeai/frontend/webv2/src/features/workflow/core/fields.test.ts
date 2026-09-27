import { DEFAULT_LORA_WEIGHT_CONFIG } from '@features/generation/settings';
import { describe, expect, it } from 'vitest';

import type { FieldInputTemplate, FieldType } from './types';

import {
  getEffectiveWorkflowFieldDescription,
  getFieldRecordId,
  getWorkflowFieldInvalidReason,
  isDirectInputField,
  isLoraFieldCollectionEntry,
  isLoraFieldWeightValid,
  isModelFieldType,
  isWorkflowCollectionItemValid,
  isWorkflowFieldValueValid,
  toLoraFieldCollectionList,
  getRandomWorkflowFieldValue,
  isShuffleableField,
  LORA_FIELD_WEIGHT_RANGE,
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
  it('honors an explicitly cleared description before falling back to a template', () => {
    const template = input({ description: 'Inherited description' });
    expect(
      getEffectiveWorkflowFieldDescription(
        { name: 'value', label: '', description: '', descriptionOverride: true },
        template
      )
    ).toBe('');
    expect(getEffectiveWorkflowFieldDescription({ name: 'value', label: '', description: '' }, template)).toBe(
      'Inherited description'
    );
  });
  it('flags missing required direct values and ignores optional fields', () => {
    expect(getWorkflowFieldInvalidReason({ isConnected: false, template: input(), value: undefined })).toBe(
      'Required value.'
    );
    expect(getWorkflowFieldInvalidReason({ isConnected: false, template: input({ required: false }), value: '' })).toBe(
      null
    );
  });

  it('accepts an empty string as a required string value', () => {
    expect(getWorkflowFieldInvalidReason({ isConnected: false, template: input(), value: '' })).toBe(null);
    expect(isWorkflowFieldValueValid(input(), '')).toBe(true);
    expect(isWorkflowFieldValueValid(input({ options: ['a'], type: single('EnumField') }), '')).toBe(false);
  });

  it('treats connected required fields as valid', () => {
    expect(
      getWorkflowFieldInvalidReason({ isConnected: true, template: input({ input: 'connection' }), value: '' })
    ).toBe(null);
  });

  it('flags missing required connections', () => {
    expect(
      getWorkflowFieldInvalidReason({ isConnected: false, template: input({ input: 'connection' }), value: undefined })
    ).toBe('Required connection.');
    expect(
      getWorkflowFieldInvalidReason({ isConnected: false, template: input({ input: 'connection' }), value: 'value' })
    ).toBe('Required connection.');
  });

  it('accepts persisted values for unsupported direct controls', () => {
    const template = input({ type: single('AnyField') });

    expect(getWorkflowFieldInvalidReason({ isConnected: false, template, value: { value: true } })).toBe(null);
    expect(getWorkflowFieldInvalidReason({ isConnected: false, template, value: undefined })).toBe(
      'Required connection.'
    );
  });

  it('validates numeric constraints', () => {
    const template = input({ maximum: 10, minimum: 1, type: single('IntegerField') });

    expect(isWorkflowFieldValueValid(template, 5)).toBe(true);
    expect(isWorkflowFieldValueValid(template, 0)).toBe(false);
    expect(isWorkflowFieldValueValid(template, 5.5)).toBe(false);
  });

  it('treats inclusive bounds, exclusive bounds, and steps exactly at their boundaries', () => {
    const inclusive = input({ maximum: 10, minimum: -2, type: single('FloatField') });

    expect(isWorkflowFieldValueValid(inclusive, -2)).toBe(true);
    expect(isWorkflowFieldValueValid(inclusive, -2.001)).toBe(false);
    expect(isWorkflowFieldValueValid(inclusive, 10)).toBe(true);
    expect(isWorkflowFieldValueValid(inclusive, 10.001)).toBe(false);

    const exclusive = input({ exclusiveMaximum: 1, exclusiveMinimum: -1, type: single('FloatField') });

    expect(isWorkflowFieldValueValid(exclusive, -1)).toBe(false);
    expect(isWorkflowFieldValueValid(exclusive, -0.999)).toBe(true);
    expect(isWorkflowFieldValueValid(exclusive, 1)).toBe(false);
    expect(isWorkflowFieldValueValid(exclusive, 0.999)).toBe(true);

    // An off-step value stays a value: it is reported, never snapped to the nearest step.
    const stepped = input({ exclusiveMinimum: -1, maximum: 1, multipleOf: 0.25, type: single('FloatField') });

    expect(isWorkflowFieldValueValid(stepped, 0.75)).toBe(true);
    expect(isWorkflowFieldValueValid(stepped, 0.7)).toBe(false);
    expect(isWorkflowFieldValueValid(stepped, -0.75)).toBe(true);
    expect(isWorkflowFieldValueValid(stepped, -1)).toBe(false);
    expect(isWorkflowFieldValueValid(stepped, -0.8)).toBe(false);
    expect(isWorkflowFieldValueValid(input({ multipleOf: 2, type: single('IntegerField') }), -4)).toBe(true);
    expect(isWorkflowFieldValueValid(input({ multipleOf: 2, type: single('IntegerField') }), -3)).toBe(false);
  });

  it('tells a cleared required number apart from a present but invalid one', () => {
    const required = input({ minimum: 0, type: single('IntegerField') });
    const optional = input({ required: false, type: single('FloatField') });

    expect(getWorkflowFieldInvalidReason({ isConnected: false, template: required, value: undefined })).toBe(
      'Required value.'
    );
    expect(getWorkflowFieldInvalidReason({ isConnected: false, template: required, value: 2.5 })).toBe(
      'Invalid value.'
    );
    expect(getWorkflowFieldInvalidReason({ isConnected: false, template: required, value: -1 })).toBe('Invalid value.');
    expect(getWorkflowFieldInvalidReason({ isConnected: false, template: optional, value: undefined })).toBe(null);
    expect(
      getWorkflowFieldInvalidReason({
        isConnected: false,
        template: input({ input: 'connection', type: single('IntegerField') }),
        value: 2.5,
      })
    ).toBe('Required connection.');
  });

  it('allows empty optional direct values but flags populated invalid optional values', () => {
    const template = input({ maximum: 10, minimum: 1, required: false, type: single('IntegerField') });

    expect(getWorkflowFieldInvalidReason({ isConnected: false, template, value: undefined })).toBe(null);
    expect(getWorkflowFieldInvalidReason({ isConnected: false, template, value: 20 })).toBe('Invalid value.');
  });

  it('validates object-backed model and image fields', () => {
    expect(isWorkflowFieldValueValid(input({ type: single('ModelIdentifierField') }), { key: 'model-key' })).toBe(true);
    expect(isWorkflowFieldValueValid(input({ type: single('ModelIdentifierField') }), {})).toBe(false);
    expect(isWorkflowFieldValueValid(input({ type: single('ImageField') }), { image_name: 'image.png' })).toBe(true);
    expect(isWorkflowFieldValueValid(input({ type: single('ImageField') }), { image_name: '' })).toBe(false);
  });

  it('validates video fields like image fields (direct-input media)', () => {
    expect(isWorkflowFieldValueValid(input({ type: single('VideoField') }), { video_name: 'clip.mp4' })).toBe(true);
    expect(isWorkflowFieldValueValid(input({ type: single('VideoField') }), { video_name: '' })).toBe(false);
    expect(isWorkflowFieldValueValid(input({ type: single('VideoField') }), {})).toBe(false);
  });

  it('validates image collections as lists of image refs and keeps video collections on the non-null check', () => {
    const videos = input({ type: { batch: false, cardinality: 'COLLECTION', name: 'VideoField' } });
    const images = input({ type: { batch: false, cardinality: 'COLLECTION', name: 'ImageField' } });

    expect(isWorkflowFieldValueValid(videos, [{ video_name: 'a.mp4' }, { video_name: 'b.mp4' }])).toBe(true);
    expect(isWorkflowFieldValueValid(videos, undefined)).toBe(false);
    expect(isWorkflowFieldValueValid(images, [{ image_name: 'a.png' }])).toBe(true);
    expect(isWorkflowFieldValueValid(images, [])).toBe(true);
    expect(isWorkflowFieldValueValid(images, [{ image_name: '' }])).toBe(false);
    expect(isWorkflowFieldValueValid(images, { image_name: 'a.png' })).toBe(false);
  });

  it('exposes image and scalar collections as direct inputs but keeps other collections connection-only', () => {
    expect(isDirectInputField(input({ type: { batch: false, cardinality: 'COLLECTION', name: 'ImageField' } }))).toBe(
      true
    );
    expect(isDirectInputField(input({ type: { batch: false, cardinality: 'COLLECTION', name: 'VideoField' } }))).toBe(
      false
    );

    for (const name of ['StringField', 'IntegerField', 'FloatField']) {
      const template = input({ type: { batch: false, cardinality: 'COLLECTION', name } });

      expect(isDirectInputField(template)).toBe(true);
      // A list has no single number to randomize.
      expect(isShuffleableField(template)).toBe(false);
    }
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

    it('enforces item counts before entries', () => {
      const floats = list('FloatField', { maxItems: 2, minItems: 1 });

      expect(reason(floats, [])).toBe('Collection is empty.');
      expect(reason(floats, [1, 2, 3])).toBe('Allows at most 2 items.');
      expect(reason(list('FloatField', { minItems: 2 }), [1])).toBe('Needs at least 2 items.');
      expect(reason(list('FloatField', { minItems: 2 }), [])).toBe('Collection is empty.');
      expect(reason(floats, [1, 2])).toBeNull();
      // Image lists share the count rules; their entries keep the image-ref check.
      expect(reason(list('ImageField', { minItems: 1 }), [])).toBe('Collection is empty.');
      expect(reason(list('ImageField'), [{ image_name: 'a.png' }, { image_name: '' }])).toBe('Item 2 is invalid.');
    });

    it('tells a missing required list apart from an empty optional one', () => {
      expect(reason(list('IntegerField'), undefined)).toBe('Required value.');
      expect(reason(list('IntegerField', { required: false }), undefined)).toBeNull();
      expect(reason(list('IntegerField', { required: false }), [null])).toBe('Item 1 is empty.');
      expect(reason(list('IntegerField', { input: 'connection' }), undefined)).toBe('Required connection.');
    });
  });

  it('accepts a LoRA collection as a list, a bare entry, or an empty list', () => {
    const loras = input({
      required: false,
      type: { batch: false, cardinality: 'SINGLE_OR_COLLECTION', name: 'LoRAField' },
    });

    expect(isWorkflowFieldValueValid(loras, [])).toBe(true);
    expect(isWorkflowFieldValueValid(loras, [LORA_ENTRY])).toBe(true);
    expect(isWorkflowFieldValueValid(loras, LORA_ENTRY)).toBe(true);
    expect(isWorkflowFieldValueValid(loras, [{ ...LORA_ENTRY, weight: 'heavy' }])).toBe(false);
    expect(getWorkflowFieldInvalidReason({ isConnected: false, template: loras, value: [] })).toBe(null);
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

  it('keeps a cleared or out-of-range LoRA weight as a readable entry that names its own reason', () => {
    const loras = input({
      required: false,
      type: { batch: false, cardinality: 'SINGLE_OR_COLLECTION', name: 'LoRAField' },
    });
    const reasonFor = (value: unknown) => getWorkflowFieldInvalidReason({ isConnected: false, template: loras, value });

    // The range is restated from the Generate LoRA config rather than imported; keep the two together.
    expect(LORA_FIELD_WEIGHT_RANGE).toEqual({
      max: DEFAULT_LORA_WEIGHT_CONFIG.numberInputMax,
      min: DEFAULT_LORA_WEIGHT_CONFIG.numberInputMin,
    });
    expect(isLoraFieldWeightValid(LORA_FIELD_WEIGHT_RANGE.max)).toBe(true);
    expect(isLoraFieldWeightValid(LORA_FIELD_WEIGHT_RANGE.min)).toBe(true);
    expect(isLoraFieldWeightValid(LORA_FIELD_WEIGHT_RANGE.max + 0.01)).toBe(false);
    expect(isLoraFieldWeightValid(null)).toBe(false);

    // A cleared weight keeps its identity (the row still renders) but cannot run.
    expect(isLoraFieldCollectionEntry({ ...LORA_ENTRY, weight: null })).toBe(true);
    expect(isWorkflowFieldValueValid(loras, [LORA_ENTRY, { ...LORA_ENTRY, weight: null }])).toBe(false);
    expect(reasonFor([LORA_ENTRY, { ...LORA_ENTRY, weight: null }])).toBe('Item 2 has no weight.');
    expect(reasonFor([{ ...LORA_ENTRY, weight: 999 }])).toBe('Item 1 needs a weight from -10 to 10.');
    expect(reasonFor({ ...LORA_ENTRY, weight: -10.5 })).toBe('Item 1 needs a weight from -10 to 10.');
    expect(reasonFor([LORA_ENTRY, { lora: { key: 'ghost' }, weight: 1 }])).toBe('Item 2 is not a readable LoRA.');
    expect(
      reasonFor([
        { ...LORA_ENTRY, weight: -10 },
        { ...LORA_ENTRY, weight: 10 },
      ])
    ).toBeNull();
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

  it('validates style preset and system prompt references by their record id', () => {
    const preset = input({ type: single('StylePresetField') });
    const prompt = input({ type: single('SystemPromptField') });

    expect(isDirectInputField(preset)).toBe(true);
    expect(isDirectInputField(prompt)).toBe(true);
    expect(isWorkflowFieldValueValid(preset, { style_preset_id: 'preset-1' })).toBe(true);
    expect(isWorkflowFieldValueValid(preset, { style_preset_id: '' })).toBe(false);
    expect(isWorkflowFieldValueValid(preset, { system_prompt_id: 'prompt-1' })).toBe(false);
    expect(isWorkflowFieldValueValid(prompt, { system_prompt_id: 'prompt-1' })).toBe(true);
    expect(isWorkflowFieldValueValid(prompt, 'prompt-1')).toBe(false);
    expect(getWorkflowFieldInvalidReason({ isConnected: false, template: preset, value: undefined })).toBe(
      'Required value.'
    );
    expect(getWorkflowFieldInvalidReason({ isConnected: false, template: preset, value: {} })).toBe('Invalid value.');
    // Core cannot see the preset list, so a stale id stays valid here and the backend rejects it on invoke.
    expect(
      getWorkflowFieldInvalidReason({ isConnected: false, template: preset, value: { style_preset_id: 'gone' } })
    ).toBeNull();
    expect(getFieldRecordId({ style_preset_id: 'preset-1' }, 'style_preset_id')).toBe('preset-1');
    expect(getFieldRecordId({ style_preset_id: ' ' }, 'style_preset_id')).toBeNull();
    expect(getFieldRecordId(undefined, 'style_preset_id')).toBeNull();
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

  it('treats empty board values as the Auto sentinel', () => {
    expect(isWorkflowFieldValueValid(input({ type: single('BoardField') }), undefined)).toBe(true);
    expect(isWorkflowFieldValueValid(input({ type: single('BoardField') }), { board_id: 'board-id' })).toBe(true);
    expect(isWorkflowFieldValueValid(input({ type: single('BoardField') }), {})).toBe(false);
  });
});

describe('LoRA collection values', () => {
  const loras = input({
    required: false,
    type: { batch: false, cardinality: 'SINGLE_OR_COLLECTION', name: 'LoRAField' },
  });

  it('gives the collection loaders an inline widget instead of a connection-only handle', () => {
    expect(isDirectInputField(loras)).toBe(true);
  });

  it('normalizes the `LoRAField | list[LoRAField]` union', () => {
    expect(toLoraFieldCollectionList([LORA_ENTRY])).toEqual([LORA_ENTRY]);
    expect(toLoraFieldCollectionList(LORA_ENTRY)).toEqual([LORA_ENTRY]);
    expect(toLoraFieldCollectionList(undefined)).toEqual([]);
    expect(toLoraFieldCollectionList(null)).toEqual([]);
  });

  it('preserves unreadable items verbatim so an edit cannot silently delete them', () => {
    // Keep unreadable collection entries so editing another row cannot destroy hand-authored data.
    const items = [LORA_ENTRY, { lora: { key: 'ghost' }, weight: 1 }, null, 'lora'];

    expect(toLoraFieldCollectionList(items)).toEqual(items);
    expect(items.filter(isLoraFieldCollectionEntry)).toEqual([LORA_ENTRY]);
  });
});

describe('workflow field type helpers', () => {
  it('matches legacy model field shape classification', () => {
    expect(isModelFieldType(single('ModelIdentifierField'))).toBe(true);
    expect(isModelFieldType(single('UNetField'))).toBe(true);
    expect(isModelFieldType(single('CLIPField'))).toBe(true);
    expect(isModelFieldType(single('ImageField'))).toBe(false);
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
  });

  it('respects exclusive bounds for integers and keeps floats unrounded', () => {
    const integer = input({ exclusiveMaximum: 5, exclusiveMinimum: 0, type: single('IntegerField') });

    expect(getRandomWorkflowFieldValue(integer, () => 0)).toBe(1);
    expect(getRandomWorkflowFieldValue(integer, () => 0.999)).toBe(4);

    const float = input({ maximum: 1, minimum: 0, type: single('FloatField') });

    expect(getRandomWorkflowFieldValue(float, () => 0.25)).toBe(0.25);

    const steppedExclusive = input({
      exclusiveMaximum: 1,
      exclusiveMinimum: 0,
      multipleOf: 0.5,
      type: single('FloatField'),
    });

    expect(getRandomWorkflowFieldValue(steppedExclusive, () => 0)).toBe(0.5);
    expect(getRandomWorkflowFieldValue(steppedExclusive, () => 0.999)).toBe(0.5);
  });

  it('keeps legacy numeric templates finite when optional constraints are absent', () => {
    const legacy = input({
      exclusiveMaximum: undefined,
      exclusiveMinimum: undefined,
      maximum: undefined,
      minimum: undefined,
      multipleOf: undefined,
      type: single('IntegerField'),
    });
    const value = getRandomWorkflowFieldValue(legacy, () => 0.5);

    expect(Number.isFinite(value)).toBe(true);
    expect(Number.isInteger(value)).toBe(true);
    expect(isWorkflowFieldValueValid(legacy, value)).toBe(true);
  });

  it('shuffles only direct numeric fields', () => {
    expect(isShuffleableField(input({ type: single('IntegerField') }))).toBe(true);
    expect(isShuffleableField(input({ type: single('FloatField') }))).toBe(true);
    expect(isShuffleableField(input({ input: 'connection', type: single('IntegerField') }))).toBe(false);
    expect(isShuffleableField(input())).toBe(false);
  });
});
