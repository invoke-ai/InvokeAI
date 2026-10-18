import type { FieldInputTemplate, WorkflowForm } from './types';

import {
  admitsWorkflowFieldStepValue,
  finiteNumberOrNull,
  getWorkflowFieldStepIndexBounds,
  toWorkflowFieldStepMultiple,
} from './fields';

/**
 * Per-element control settings an author gives a form field: which control renders it and, for numbers, the
 * bounds the form offers. They are stored in the element's legacy settings bag (`NodeFieldFormElement.data.settings`)
 * in the shape the v6 editor wrote, so a workflow authored in either editor opens the same in the other.
 */

export type NumberFieldComponent = 'number-input' | 'slider' | 'number-input-and-slider';
export type StringFieldComponent = 'input' | 'textarea' | 'dropdown';

export interface NodeFieldStringOption {
  label: string;
  value: string;
}

export interface NodeFieldNumberSettings {
  kind: 'number';
  component: NumberFieldComponent;
  /**
   * Authored bounds and step; each applies only where it tightens the invocation template's own constraint without
   * leaving its range.
   */
  min?: number;
  max?: number;
  step?: number;
}

export interface NodeFieldStringSettings {
  kind: 'string';
  component: StringFieldComponent;
  /** The choices a dropdown offers; ignored by the other controls. */
  options: NodeFieldStringOption[];
}

export type NodeFieldFormSettings = NodeFieldNumberSettings | NodeFieldStringSettings;

export const NUMBER_FIELD_COMPONENTS: readonly NumberFieldComponent[] = [
  'number-input',
  'slider',
  'number-input-and-slider',
];
export const STRING_FIELD_COMPONENTS: readonly StringFieldComponent[] = ['input', 'textarea', 'dropdown'];

// The v6 editor tags each bag with the field kind it was authored for.
const LEGACY_SETTINGS_TYPES = {
  FloatField: 'float-field-config',
  IntegerField: 'integer-field-config',
  StringField: 'string-field-config',
} as const;

type SettingsFieldTypeName = keyof typeof LEGACY_SETTINGS_TYPES;

const isSettingsFieldTypeName = (name: string): name is SettingsFieldTypeName => name in LEGACY_SETTINGS_TYPES;

/** True for the scalar fields a form element can carry control settings for. */
export const hasNodeFieldFormSettings = (template: FieldInputTemplate): boolean =>
  template.type.cardinality === 'SINGLE' && isSettingsFieldTypeName(template.type.name);

const finiteOrUndefined = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isFinite(value) ? value : undefined;

const isNumberComponent = (value: unknown): value is NumberFieldComponent =>
  typeof value === 'string' && (NUMBER_FIELD_COMPONENTS as readonly string[]).includes(value);

const isStringComponent = (value: unknown): value is StringFieldComponent =>
  typeof value === 'string' && (STRING_FIELD_COMPONENTS as readonly string[]).includes(value);

const toOptions = (value: unknown): NodeFieldStringOption[] =>
  Array.isArray(value)
    ? value.flatMap((option) =>
        typeof option === 'object' && option !== null
          ? [
              {
                label:
                  typeof (option as { label?: unknown }).label === 'string' ? (option as { label: string }).label : '',
                value:
                  typeof (option as { value?: unknown }).value === 'string' ? (option as { value: string }).value : '',
              },
            ]
          : []
      )
    : [];

/** The control settings the template's own metadata implies, used until an author chooses otherwise. */
export const getDefaultNodeFieldFormSettings = (template: FieldInputTemplate): NodeFieldFormSettings | null => {
  if (!hasNodeFieldFormSettings(template)) {
    return null;
  }

  return template.type.name === 'StringField'
    ? { component: template.uiComponent === 'textarea' ? 'textarea' : 'input', kind: 'string', options: [] }
    : { component: 'number-input', kind: 'number' };
};

/**
 * The authored settings of a form element for `template`, or null when the element has none for this kind of
 * field. A bag tagged for another field kind (the field's type changed under it) is ignored rather than misread;
 * a bag with unusable bounds keeps its control choice and drops the bounds.
 */
export const getNodeFieldFormSettings = (
  settings: Record<string, unknown> | undefined,
  template: FieldInputTemplate
): NodeFieldFormSettings | null => {
  if (!settings || !hasNodeFieldFormSettings(template)) {
    return null;
  }

  const typeName = template.type.name as SettingsFieldTypeName;

  if (settings.type !== undefined && settings.type !== LEGACY_SETTINGS_TYPES[typeName]) {
    return null;
  }

  if (typeName === 'StringField') {
    const component = isStringComponent(settings.component) ? settings.component : 'input';

    return { component, kind: 'string', options: component === 'dropdown' ? toOptions(settings.options) : [] };
  }

  const component = isNumberComponent(settings.component) ? settings.component : 'number-input';
  const min = finiteOrUndefined(settings.min);
  const max = finiteOrUndefined(settings.max);
  const step = finiteOrUndefined(settings.step);

  // Bounds are kept as authored, even inverted, so the builder edits what is stored; `applyNodeFieldFormSettings`
  // decides what the form honours.
  return {
    component,
    kind: 'number',
    ...(min !== undefined ? { min } : {}),
    ...(max !== undefined ? { max } : {}),
    ...(step !== undefined && step > 0 ? { step } : {}),
  };
};

/** Authored bounds that cannot form a range; the form ignores both until the author fixes one. */
const hasInvertedNodeFieldBounds = (settings: NodeFieldFormSettings | null): boolean =>
  settings?.kind === 'number' &&
  settings.min !== undefined &&
  settings.max !== undefined &&
  settings.min > settings.max;

// The bag keys these settings own; a cleared one is left out rather than carried over from the stored bag.
const OWNED_SETTINGS_KEYS = new Set(['component', 'max', 'min', 'options', 'showShuffle', 'step', 'type']);

/**
 * The settings bag to store for `settings` on a field of `template`, in the legacy shape: tagged with the field
 * kind, and carrying `showShuffle` on number fields as the v6 editor does. Keys of `stored` these settings do not
 * own (another editor's) are kept.
 */
export const serializeNodeFieldFormSettings = (
  settings: NodeFieldFormSettings,
  template: FieldInputTemplate,
  showShuffle: boolean,
  stored?: Record<string, unknown>
): Record<string, unknown> => {
  const type = isSettingsFieldTypeName(template.type.name) ? LEGACY_SETTINGS_TYPES[template.type.name] : undefined;
  const unowned = stored
    ? Object.fromEntries(Object.entries(stored).filter(([key]) => !OWNED_SETTINGS_KEYS.has(key)))
    : {};

  if (settings.kind === 'string') {
    return {
      ...unowned,
      component: settings.component,
      type,
      ...(settings.component === 'dropdown' ? { options: settings.options.map((option) => ({ ...option })) } : {}),
    };
  }

  return {
    ...unowned,
    component: settings.component,
    showShuffle,
    type,
    ...(settings.min !== undefined ? { min: settings.min } : {}),
    ...(settings.max !== undefined ? { max: settings.max } : {}),
    ...(settings.step !== undefined ? { step: settings.step } : {}),
  };
};

const isMultipleOf = (step: number, base: number): boolean => {
  const ratio = step / base;

  return Math.abs(ratio - Math.round(ratio)) < 1e-9;
};

// Whether an authored bound lies inside the template's opposite end. One beyond it (a minimum above the template's
// maximum) would leave no value in range, so the form ignores it as it does an inverted pair.
const isMinimumInsideTemplate = (template: FieldInputTemplate, min: number): boolean => {
  const maximum = finiteNumberOrNull(template.maximum);
  const exclusiveMaximum = finiteNumberOrNull(template.exclusiveMaximum);

  return maximum !== null ? min <= maximum : exclusiveMaximum === null || min < exclusiveMaximum;
};

const isMaximumInsideTemplate = (template: FieldInputTemplate, max: number): boolean => {
  const minimum = finiteNumberOrNull(template.minimum);
  const exclusiveMinimum = finiteNumberOrNull(template.exclusiveMinimum);

  return minimum !== null ? max >= minimum : exclusiveMinimum === null || max > exclusiveMinimum;
};

/** A whole step for an integer field, and a multiple of the node's own step when it has one. */
const isStepAcceptedByTemplate = (template: FieldInputTemplate, step: number): boolean =>
  (template.type.name !== 'IntegerField' || Number.isInteger(step)) &&
  (template.multipleOf === null || template.multipleOf <= 0 || isMultipleOf(step, template.multipleOf));

/**
 * The template narrowed by the authored number settings: a bound applies only where it tightens the template's own
 * end and stays inside its other one (an inverted pair not at all), and a step only when the template has none or
 * the step keeps every value a multiple of the template's. Validation, shuffle and the control all read this
 * template, so the form never offers a value the node would reject. Ignored settings stay stored for the author to
 * correct; `getNodeFieldSettingsIssue` says why.
 */
export const applyNodeFieldFormSettings = (
  template: FieldInputTemplate,
  settings: NodeFieldFormSettings | null
): FieldInputTemplate => {
  if (!settings || settings.kind !== 'number') {
    return template;
  }

  const inverted = hasInvertedNodeFieldBounds(settings);
  let next = template;
  const patch = (fields: Partial<FieldInputTemplate>) => {
    next = next === template ? { ...template, ...fields } : { ...next, ...fields };
  };

  if (settings.min !== undefined && !inverted && isMinimumInsideTemplate(template, settings.min)) {
    if (template.minimum !== null) {
      if (settings.min > template.minimum) {
        patch({ minimum: settings.min });
      }
    } else if (template.exclusiveMinimum === null || settings.min > template.exclusiveMinimum) {
      patch({ exclusiveMinimum: null, minimum: settings.min });
    }
  }

  if (settings.max !== undefined && !inverted && isMaximumInsideTemplate(template, settings.max)) {
    if (template.maximum !== null) {
      if (settings.max < template.maximum) {
        patch({ maximum: settings.max });
      }
    } else if (template.exclusiveMaximum === null || settings.max < template.exclusiveMaximum) {
      patch({ exclusiveMaximum: null, maximum: settings.max });
    }
  }

  if (settings.step !== undefined && isStepAcceptedByTemplate(template, settings.step)) {
    patch({ multipleOf: settings.step });
  }

  return next;
};

export interface NodeFieldSliderRange {
  max: number;
  min: number;
  step: number;
}

// The track step of a float field that accepts any value in range, as in v6.
const FLOAT_SLIDER_STEP = 0.01;

/**
 * The range a slider can show for `template` (already narrowed by `applyNodeFieldFormSettings`), or null when it
 * has no track: an end is unbounded, or fewer than two stops would pass validation. The host then keeps the number
 * input.
 *
 * Validation accepts only multiples of the step counted from zero (every integer, for an integer field without
 * one), so the track runs between the first and last multiples inside the bounds rather than the bounds
 * themselves; a slider stepping by 8 between 1 and 100 offers 8…96, not 1…97.
 */
export const getNodeFieldSliderRange = (template: FieldInputTemplate): NodeFieldSliderRange | null => {
  const multipleOf = finiteNumberOrNull(template.multipleOf);
  const step = multipleOf !== null && multipleOf > 0 ? multipleOf : template.type.name === 'IntegerField' ? 1 : null;

  if (step !== null) {
    const { first, last } = getWorkflowFieldStepIndexBounds(template, step);

    if (first === null || last === null || first >= last) {
      return null;
    }

    return { max: toWorkflowFieldStepMultiple(last, step), min: toWorkflowFieldStepMultiple(first, step), step };
  }

  const exclusiveMinimum = finiteNumberOrNull(template.exclusiveMinimum);
  const exclusiveMaximum = finiteNumberOrNull(template.exclusiveMaximum);
  const min =
    finiteNumberOrNull(template.minimum) ?? (exclusiveMinimum !== null ? exclusiveMinimum + FLOAT_SLIDER_STEP : null);
  const max =
    finiteNumberOrNull(template.maximum) ?? (exclusiveMaximum !== null ? exclusiveMaximum - FLOAT_SLIDER_STEP : null);

  if (min === null || max === null || min >= max) {
    return null;
  }

  return { max, min, step: FLOAT_SLIDER_STEP };
};

export type NodeFieldSettingsIssue = 'inverted' | 'outOfRange' | 'stepIgnored' | 'noValue' | 'noTrack';

/**
 * Why the form cannot honour authored number settings as written, in the order an author would fix them; null when
 * it can. `inverted` and `outOfRange` bounds and a `stepIgnored` step are ignored by `applyNodeFieldFormSettings`;
 * with `noValue` no value passes validation, so Invoke stays blocked; with `noTrack` a slider control falls back to
 * the number input.
 */
export const getNodeFieldSettingsIssue = (
  template: FieldInputTemplate,
  settings: NodeFieldFormSettings | null
): NodeFieldSettingsIssue | null => {
  if (settings?.kind !== 'number') {
    return null;
  }

  if (hasInvertedNodeFieldBounds(settings)) {
    return 'inverted';
  }

  if (
    (settings.min !== undefined && !isMinimumInsideTemplate(template, settings.min)) ||
    (settings.max !== undefined && !isMaximumInsideTemplate(template, settings.max))
  ) {
    return 'outOfRange';
  }

  if (settings.step !== undefined && !isStepAcceptedByTemplate(template, settings.step)) {
    return 'stepIgnored';
  }

  const effectiveTemplate = applyNodeFieldFormSettings(template, settings);

  if (!admitsWorkflowFieldStepValue(effectiveTemplate)) {
    return 'noValue';
  }

  return settings.component !== 'number-input' && getNodeFieldSliderRange(effectiveTemplate) === null
    ? 'noTrack'
    : null;
};

/**
 * The template a node field is validated against: the invocation template narrowed by the settings of the form
 * element exposing it, or the template itself when the field is not on the form. Invoke readiness and the builder's
 * invalid markers read this so they agree with the control.
 */
export const getFormFieldTemplate = (
  form: WorkflowForm,
  nodeId: string,
  fieldName: string,
  template: FieldInputTemplate
): FieldInputTemplate => {
  for (const element of Object.values(form.elements)) {
    if (
      element.type === 'node-field' &&
      element.data.fieldIdentifier.nodeId === nodeId &&
      element.data.fieldIdentifier.fieldName === fieldName
    ) {
      return applyNodeFieldFormSettings(template, getNodeFieldFormSettings(element.data.settings, template));
    }
  }

  return template;
};
