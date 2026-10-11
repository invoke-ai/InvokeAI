import { describe, expect, it } from 'vitest';

import type { FieldInputTemplate } from './types';

import {
  applyNodeFieldFormSettings,
  getDefaultNodeFieldFormSettings,
  getNodeFieldFormSettings,
  getNodeFieldSliderRange,
  serializeNodeFieldFormSettings,
} from './formFieldSettings';

const template = (
  typeName: 'FloatField' | 'IntegerField' | 'StringField',
  overrides: Partial<FieldInputTemplate> = {}
): FieldInputTemplate => ({
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
  type: { batch: false, cardinality: 'SINGLE', name: typeName },
  uiChoiceLabels: null,
  uiComponent: null,
  uiHidden: false,
  uiModelBase: null,
  uiModelFormat: null,
  uiModelType: null,
  uiOrder: null,
  ...overrides,
});

describe('getNodeFieldFormSettings', () => {
  it('reads a legacy integer bag with its control and bounds', () => {
    const settings = getNodeFieldFormSettings(
      { component: 'number-input-and-slider', max: 50, min: 1, showShuffle: true, type: 'integer-field-config' },
      template('IntegerField')
    );

    expect(settings).toEqual({ component: 'number-input-and-slider', kind: 'number', max: 50, min: 1 });
  });

  it('reads a legacy dropdown bag and tolerates malformed options', () => {
    const settings = getNodeFieldFormSettings(
      {
        component: 'dropdown',
        options: [{ label: 'Fast', value: 'fast' }, { value: 'slow' }, 'junk', null],
        type: 'string-field-config',
      },
      template('StringField')
    );

    expect(settings).toEqual({
      component: 'dropdown',
      kind: 'string',
      options: [
        { label: 'Fast', value: 'fast' },
        { label: '', value: 'slow' },
      ],
    });
  });

  it('ignores a bag tagged for another field kind and bags on fields that cannot carry one', () => {
    expect(
      getNodeFieldFormSettings({ component: 'slider', type: 'float-field-config' }, template('IntegerField'))
    ).toBeNull();
    expect(
      getNodeFieldFormSettings(
        { component: 'slider', type: 'integer-field-config' },
        template('IntegerField', { type: { batch: false, cardinality: 'COLLECTION', name: 'IntegerField' } })
      )
    ).toBeNull();
    expect(getNodeFieldFormSettings(undefined, template('IntegerField'))).toBeNull();
  });

  it('keeps the control but drops bounds that cannot form a range', () => {
    expect(
      getNodeFieldFormSettings({ component: 'slider', max: 1, min: 10, step: -2 }, template('FloatField'))
    ).toEqual({ component: 'slider', kind: 'number' });
  });

  it('falls back to the plain control for an unknown component name', () => {
    expect(getNodeFieldFormSettings({ component: 'knob' }, template('FloatField'))).toEqual({
      component: 'number-input',
      kind: 'number',
    });
    expect(getNodeFieldFormSettings({ component: 'rich-text' }, template('StringField'))).toEqual({
      component: 'input',
      kind: 'string',
      options: [],
    });
  });
});

describe('getDefaultNodeFieldFormSettings', () => {
  it('starts a prompt-like string field as a text area and numbers as inputs', () => {
    expect(getDefaultNodeFieldFormSettings(template('StringField', { uiComponent: 'textarea' }))).toEqual({
      component: 'textarea',
      kind: 'string',
      options: [],
    });
    expect(getDefaultNodeFieldFormSettings(template('FloatField'))).toEqual({
      component: 'number-input',
      kind: 'number',
    });
  });
});

describe('serializeNodeFieldFormSettings', () => {
  it('writes the legacy shape the v6 editor reads back, round-tripping through the reader', () => {
    const numberBag = serializeNodeFieldFormSettings(
      { component: 'slider', kind: 'number', max: 8, min: 2, step: 2 },
      template('IntegerField'),
      true
    );

    expect(numberBag).toEqual({
      component: 'slider',
      max: 8,
      min: 2,
      showShuffle: true,
      step: 2,
      type: 'integer-field-config',
    });
    expect(getNodeFieldFormSettings(numberBag, template('IntegerField'))).toEqual({
      component: 'slider',
      kind: 'number',
      max: 8,
      min: 2,
      step: 2,
    });

    const stringBag = serializeNodeFieldFormSettings(
      { component: 'dropdown', kind: 'string', options: [{ label: 'A', value: 'a' }] },
      template('StringField'),
      false
    );

    expect(stringBag).toEqual({
      component: 'dropdown',
      options: [{ label: 'A', value: 'a' }],
      type: 'string-field-config',
    });
  });

  it('does not keep dropdown options on a field switched back to a text input', () => {
    expect(
      serializeNodeFieldFormSettings(
        { component: 'input', kind: 'string', options: [] },
        template('StringField'),
        false
      )
    ).toEqual({ component: 'input', type: 'string-field-config' });
  });
});

describe('applyNodeFieldFormSettings', () => {
  it('narrows the template to the authored bounds but never widens them', () => {
    const base = template('IntegerField', { maximum: 100, minimum: 0 });

    expect(applyNodeFieldFormSettings(base, { component: 'slider', kind: 'number', max: 200, min: 10 })).toMatchObject({
      maximum: 100,
      minimum: 10,
    });
  });

  it('replaces an exclusive template bound only with a tighter inclusive one', () => {
    const base = template('FloatField', { exclusiveMinimum: 0 });

    expect(applyNodeFieldFormSettings(base, { component: 'slider', kind: 'number', min: 0.5 })).toMatchObject({
      exclusiveMinimum: null,
      minimum: 0.5,
    });
    expect(applyNodeFieldFormSettings(base, { component: 'slider', kind: 'number', min: -1 })).toBe(base);
  });

  it('applies a step only where it keeps every value valid for the node', () => {
    const free = template('FloatField');
    const eights = template('IntegerField', { multipleOf: 8 });

    expect(applyNodeFieldFormSettings(free, { component: 'slider', kind: 'number', step: 0.25 }).multipleOf).toBe(0.25);
    expect(applyNodeFieldFormSettings(eights, { component: 'slider', kind: 'number', step: 64 }).multipleOf).toBe(64);
    expect(applyNodeFieldFormSettings(eights, { component: 'slider', kind: 'number', step: 12 })).toBe(eights);
  });

  it('returns the template itself for string settings and unchanged bounds', () => {
    const base = template('StringField');

    expect(applyNodeFieldFormSettings(base, { component: 'dropdown', kind: 'string', options: [] })).toBe(base);
    expect(applyNodeFieldFormSettings(base, null)).toBe(base);
  });
});

describe('getNodeFieldSliderRange', () => {
  it('derives a closed range with the template step, stepping inside exclusive ends', () => {
    expect(getNodeFieldSliderRange(template('IntegerField', { maximum: 10, minimum: 1 }))).toEqual({
      max: 10,
      min: 1,
      step: 1,
    });
    expect(
      getNodeFieldSliderRange(template('FloatField', { exclusiveMaximum: 1, exclusiveMinimum: 0, multipleOf: 0.1 }))
    ).toEqual({ max: 0.9, min: 0.1, step: 0.1 });
  });

  it('has no range while an end is open', () => {
    expect(getNodeFieldSliderRange(template('IntegerField', { minimum: 0 }))).toBeNull();
  });
});
