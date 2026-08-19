import { describe, expect, it } from 'vitest';

import type { FieldInputTemplate, FieldType } from './types';

import { getWorkflowFieldInvalidReason, isModelFieldType, isWorkflowFieldValueValid } from './fields';

const single = (name: string): FieldType => ({ batch: false, cardinality: 'SINGLE', name });

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
  it('flags empty required direct values and ignores optional fields', () => {
    expect(getWorkflowFieldInvalidReason({ isConnected: false, template: input(), value: '' })).toBe('Required value.');
    expect(getWorkflowFieldInvalidReason({ isConnected: false, template: input({ required: false }), value: '' })).toBe(
      null
    );
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

  it('keeps COLLECTION media values on the generic non-null check (persisted arrays stay valid)', () => {
    const videos = input({ type: { batch: false, cardinality: 'COLLECTION', name: 'VideoField' } });
    const images = input({ type: { batch: false, cardinality: 'COLLECTION', name: 'ImageField' } });

    expect(isWorkflowFieldValueValid(videos, [{ video_name: 'a.mp4' }, { video_name: 'b.mp4' }])).toBe(true);
    expect(isWorkflowFieldValueValid(videos, undefined)).toBe(false);
    expect(isWorkflowFieldValueValid(images, [{ image_name: 'a.png' }])).toBe(true);
  });

  it('treats empty board values as the Auto sentinel', () => {
    expect(isWorkflowFieldValueValid(input({ type: single('BoardField') }), undefined)).toBe(true);
    expect(isWorkflowFieldValueValid(input({ type: single('BoardField') }), { board_id: 'board-id' })).toBe(true);
    expect(isWorkflowFieldValueValid(input({ type: single('BoardField') }), {})).toBe(false);
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
