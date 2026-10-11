import type { FieldInputTemplate } from './types';

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
  /** Authored bounds and step; each applies only where it tightens the invocation template's own constraint. */
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
  const boundsUsable = min === undefined || max === undefined || min <= max;

  return {
    component,
    kind: 'number',
    ...(boundsUsable && min !== undefined ? { min } : {}),
    ...(boundsUsable && max !== undefined ? { max } : {}),
    ...(step !== undefined && step > 0 ? { step } : {}),
  };
};

/**
 * The settings bag to store for `settings` on a field of `template`, in the legacy shape: tagged with the field
 * kind, and carrying `showShuffle` on number fields as the v6 editor does.
 */
export const serializeNodeFieldFormSettings = (
  settings: NodeFieldFormSettings,
  template: FieldInputTemplate,
  showShuffle: boolean
): Record<string, unknown> => {
  const type = isSettingsFieldTypeName(template.type.name) ? LEGACY_SETTINGS_TYPES[template.type.name] : undefined;

  if (settings.kind === 'string') {
    return {
      component: settings.component,
      type,
      ...(settings.component === 'dropdown' ? { options: settings.options.map((option) => ({ ...option })) } : {}),
    };
  }

  return {
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

/**
 * The template narrowed by the authored number settings: a bound applies only inside the template's own range, and
 * a step only when the template has none or the step keeps every value a multiple of the template's. Validation,
 * shuffle and the control all read this template, so the form can never offer a value the node would reject.
 */
export const applyNodeFieldFormSettings = (
  template: FieldInputTemplate,
  settings: NodeFieldFormSettings | null
): FieldInputTemplate => {
  if (!settings || settings.kind !== 'number') {
    return template;
  }

  let next = template;
  const patch = (fields: Partial<FieldInputTemplate>) => {
    next = next === template ? { ...template, ...fields } : { ...next, ...fields };
  };

  if (settings.min !== undefined) {
    if (template.minimum !== null) {
      if (settings.min > template.minimum) {
        patch({ minimum: settings.min });
      }
    } else if (template.exclusiveMinimum === null || settings.min > template.exclusiveMinimum) {
      patch({ exclusiveMinimum: null, minimum: settings.min });
    }
  }

  if (settings.max !== undefined) {
    if (template.maximum !== null) {
      if (settings.max < template.maximum) {
        patch({ maximum: settings.max });
      }
    } else if (template.exclusiveMaximum === null || settings.max < template.exclusiveMaximum) {
      patch({ exclusiveMaximum: null, maximum: settings.max });
    }
  }

  if (
    settings.step !== undefined &&
    (template.multipleOf === null || template.multipleOf <= 0 || isMultipleOf(settings.step, template.multipleOf))
  ) {
    patch({ multipleOf: settings.step });
  }

  return next;
};

export interface NodeFieldSliderRange {
  max: number;
  min: number;
  step: number;
}

/**
 * The range a slider can show for `template` (already narrowed by `applyNodeFieldFormSettings`), or null when an
 * end is unbounded: a slider over an open range has no meaningful track, so the host keeps the number input.
 */
export const getNodeFieldSliderRange = (template: FieldInputTemplate): NodeFieldSliderRange | null => {
  const isInteger = template.type.name === 'IntegerField';
  const step = template.multipleOf !== null && template.multipleOf > 0 ? template.multipleOf : isInteger ? 1 : 0.01;
  const min = template.minimum ?? (template.exclusiveMinimum !== null ? template.exclusiveMinimum + step : null);
  const max = template.maximum ?? (template.exclusiveMaximum !== null ? template.exclusiveMaximum - step : null);

  if (min === null || max === null || !Number.isFinite(min) || !Number.isFinite(max) || min > max) {
    return null;
  }

  return { max, min, step };
};
