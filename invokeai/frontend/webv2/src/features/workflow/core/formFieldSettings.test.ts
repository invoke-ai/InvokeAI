import { describe, expect, it } from 'vitest';

import type { FieldInputTemplate } from './types';

import {
  applyNodeFieldFormSettings,
  getDefaultNodeFieldFormSettings,
  getNodeFieldFormSettings,
  getNodeFieldSettingsIssue,
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

  it('keeps inverted bounds as authored and drops a non-positive step', () => {
    expect(
      getNodeFieldFormSettings({ component: 'slider', max: 1, min: 10, step: -2 }, template('FloatField'))
    ).toEqual({ component: 'slider', kind: 'number', max: 1, min: 10 });
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

  it('keeps keys of the stored bag it does not own and drops owned ones the edit cleared', () => {
    expect(
      serializeNodeFieldFormSettings({ component: 'slider', kind: 'number', max: 8 }, template('IntegerField'), false, {
        component: 'slider',
        max: 8,
        min: 2,
        showShuffle: true,
        sliderMarks: [2, 8],
        type: 'integer-field-config',
      })
    ).toEqual({ component: 'slider', max: 8, showShuffle: false, sliderMarks: [2, 8], type: 'integer-field-config' });
    expect(
      serializeNodeFieldFormSettings(
        { component: 'input', kind: 'string', options: [] },
        template('StringField'),
        false,
        {
          component: 'dropdown',
          options: [{ label: 'A', value: 'a' }],
          placeholder: 'Pick one',
          type: 'string-field-config',
        }
      )
    ).toEqual({ component: 'input', placeholder: 'Pick one', type: 'string-field-config' });
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
    // An integer field cannot take a fractional step: the slider would commit values the node rejects.
    const whole = template('IntegerField');
    expect(applyNodeFieldFormSettings(whole, { component: 'slider', kind: 'number', step: 0.5 })).toBe(whole);
  });

  it('ignores an inverted pair of bounds but still applies the step', () => {
    const base = template('IntegerField', { maximum: 100, minimum: 0 });

    expect(applyNodeFieldFormSettings(base, { component: 'slider', kind: 'number', max: 5, min: 10 })).toBe(base);
    expect(
      applyNodeFieldFormSettings(base, { component: 'slider', kind: 'number', max: 5, min: 10, step: 5 })
    ).toMatchObject({ maximum: 100, minimum: 0, multipleOf: 5 });
  });

  it("ignores a bound beyond the template's other end, which would leave no valid value", () => {
    const closed = template('IntegerField', { maximum: 10, minimum: 0 });

    expect(applyNodeFieldFormSettings(closed, { component: 'slider', kind: 'number', min: 20 })).toBe(closed);
    expect(applyNodeFieldFormSettings(closed, { component: 'slider', kind: 'number', max: -5 })).toBe(closed);
    // Either end may be reached exactly, which leaves a single valid value.
    expect(applyNodeFieldFormSettings(closed, { component: 'slider', kind: 'number', min: 10 })).toMatchObject({
      maximum: 10,
      minimum: 10,
    });
    // An exclusive end admits values up to, not at, the bound.
    const open = template('FloatField', { exclusiveMaximum: 10, exclusiveMinimum: 0 });

    expect(applyNodeFieldFormSettings(open, { component: 'slider', kind: 'number', min: 10 })).toBe(open);
    expect(applyNodeFieldFormSettings(open, { component: 'slider', kind: 'number', max: 0 })).toBe(open);
    expect(applyNodeFieldFormSettings(open, { component: 'slider', kind: 'number', max: 9.5, min: 0.5 })).toMatchObject(
      { exclusiveMaximum: null, exclusiveMinimum: null, maximum: 9.5, minimum: 0.5 }
    );
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
    expect(getNodeFieldSliderRange(template('FloatField', { exclusiveMaximum: 1, exclusiveMinimum: 0 }))).toEqual({
      max: 0.99,
      min: 0.01,
      step: 0.01,
    });
  });

  it('ends the track on the step multiples validation accepts, not on the bounds', () => {
    // Every stop must pass `isWorkflowFieldValueValid`, which counts multiples from zero.
    expect(getNodeFieldSliderRange(template('IntegerField', { maximum: 100, minimum: 1, multipleOf: 8 }))).toEqual({
      max: 96,
      min: 8,
      step: 8,
    });
    expect(getNodeFieldSliderRange(template('IntegerField', { maximum: 9.5, minimum: 1.5 }))).toEqual({
      max: 9,
      min: 2,
      step: 1,
    });
    expect(getNodeFieldSliderRange(template('FloatField', { maximum: 0.95, minimum: 0.05, multipleOf: 0.1 }))).toEqual({
      max: 0.9,
      min: 0.1,
      step: 0.1,
    });
  });

  it('keeps the precision of a step written in exponent form', () => {
    // String(1e-7) is "1e-7": rounding to its digits after the point alone would collapse every stop to 0.
    expect(getNodeFieldSliderRange(template('FloatField', { maximum: 9e-7, minimum: 1e-7, multipleOf: 1e-7 }))).toEqual(
      { max: 9e-7, min: 1e-7, step: 1e-7 }
    );
    expect(getNodeFieldSliderRange(template('FloatField', { maximum: 3e-7, minimum: 0, multipleOf: 1.5e-7 }))).toEqual({
      max: 3e-7,
      min: 0,
      step: 1.5e-7,
    });
  });

  it('keeps a bound that is itself a multiple despite float noise, and steps past an exclusive one', () => {
    // 1.1 / 0.1 and 0.3 / 0.1 are not integers in floating point; validation tolerates that and so must the track.
    expect(getNodeFieldSliderRange(template('FloatField', { maximum: 1.1, minimum: 0.3, multipleOf: 0.1 }))).toEqual({
      max: 1.1,
      min: 0.3,
      step: 0.1,
    });
    expect(
      getNodeFieldSliderRange(template('FloatField', { exclusiveMaximum: 1.1, exclusiveMinimum: 0.3, multipleOf: 0.1 }))
    ).toEqual({ max: 1, min: 0.4, step: 0.1 });
  });

  it('has no range while an end is open or no stop would be valid', () => {
    expect(getNodeFieldSliderRange(template('IntegerField', { minimum: 0 }))).toBeNull();
    expect(getNodeFieldSliderRange(template('IntegerField', { maximum: 7, minimum: 1, multipleOf: 8 }))).toBeNull();
    expect(getNodeFieldSliderRange(template('IntegerField', { maximum: 2.9, minimum: 2.1 }))).toBeNull();
  });

  it('has no range when the bounds admit a single stop, which a track could not move between', () => {
    expect(getNodeFieldSliderRange(template('IntegerField', { maximum: 8, minimum: 8 }))).toBeNull();
    expect(getNodeFieldSliderRange(template('IntegerField', { maximum: 12, minimum: 5, multipleOf: 8 }))).toBeNull();
    expect(getNodeFieldSliderRange(template('FloatField', { maximum: 0.5, minimum: 0.5 }))).toBeNull();
  });
});

describe('getNodeFieldSettingsIssue', () => {
  const closed = template('IntegerField', { maximum: 100, minimum: 0 });

  it('names the problem the author has to fix first', () => {
    expect(getNodeFieldSettingsIssue(closed, { component: 'number-input', kind: 'number', max: 5, min: 10 })).toBe(
      'inverted'
    );
    expect(getNodeFieldSettingsIssue(closed, { component: 'number-input', kind: 'number', min: 200 })).toBe(
      'outOfRange'
    );
    // A fractional step on an integer field, or one off the node's own step, is ignored rather than applied.
    expect(getNodeFieldSettingsIssue(closed, { component: 'number-input', kind: 'number', step: 2.5 })).toBe(
      'stepIgnored'
    );
    const halves = template('FloatField', { maximum: 10, minimum: 0, multipleOf: 0.5 });
    expect(getNodeFieldSettingsIssue(halves, { component: 'number-input', kind: 'number', step: 0.3 })).toBe(
      'stepIgnored'
    );
    expect(getNodeFieldSettingsIssue(halves, { component: 'number-input', kind: 'number', step: 1.5 })).toBeNull();
    // Bounds and step are each valid, but no multiple of 8 lies between 1 and 7: every value fails validation.
    expect(
      getNodeFieldSettingsIssue(closed, { component: 'number-input', kind: 'number', max: 7, min: 1, step: 8 })
    ).toBe('noValue');
    expect(getNodeFieldSettingsIssue(closed, { component: 'slider', kind: 'number', max: 8, min: 8 })).toBe('noTrack');
  });

  it('reports nothing for settings the form honours, including an input on an open-ended field', () => {
    expect(getNodeFieldSettingsIssue(closed, { component: 'slider', kind: 'number', max: 50, min: 10 })).toBeNull();
    // Bounds that do not tighten the template are not applied, but they leave the range intact.
    expect(getNodeFieldSettingsIssue(closed, { component: 'slider', kind: 'number', max: 500, min: -5 })).toBeNull();
    expect(
      getNodeFieldSettingsIssue(template('IntegerField', { minimum: 0 }), { component: 'number-input', kind: 'number' })
    ).toBeNull();
    expect(
      getNodeFieldSettingsIssue(template('StringField'), { component: 'dropdown', kind: 'string', options: [] })
    ).toBeNull();
  });
});
