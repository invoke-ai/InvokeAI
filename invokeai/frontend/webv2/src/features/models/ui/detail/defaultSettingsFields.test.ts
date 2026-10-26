import type { ModelConfig } from '@features/models/core/types';
import type { ModelSchedulerPolicy } from '@features/models/ui/ModelsUiContext';

import { describe, expect, it } from 'vitest';

import { getFieldsForModel, validateDefaults, type SchedulerFieldRules } from './defaultSettingsFields';

type FieldsModel = Pick<ModelConfig, 'base' | 'default_settings' | 'type'>;

const model = (base: string, type: string, defaultSettings: ModelConfig['default_settings'] = null): FieldsModel =>
  ({ base, default_settings: defaultSettings, type }) as FieldsModel;

const ANIMA_POLICY: ModelSchedulerPolicy = {
  appliesToGraph: true,
  defaultValue: 'euler',
  options: [
    { label: 'Euler', value: 'euler' },
    { label: 'ER-SDE', value: 'er_sde' },
  ],
};
const IGNORED_POLICY: ModelSchedulerPolicy = { ...ANIMA_POLICY, appliesToGraph: false };

const fieldKeys = (
  candidate: FieldsModel,
  fp8StorageSupported = true,
  schedulerRules: SchedulerFieldRules | null = ANIMA_POLICY
): string[] => getFieldsForModel(candidate, fp8StorageSupported, schedulerRules).map((field) => String(field.key));

const schedulerField = (candidate: FieldsModel, policy: ModelSchedulerPolicy) =>
  getFieldsForModel(candidate, false, policy).find((field) => field.key === 'scheduler');

describe('getFieldsForModel', () => {
  it('appends fp8_storage to the main-model fields without disturbing the existing ones', () => {
    const keys = fieldKeys(model('sdxl', 'main'));

    expect(keys).toContain('fp8_storage');
    expect(keys.slice(0, -1)).toEqual([
      'vae',
      'scheduler',
      'steps',
      'cfg_scale',
      'cfg_rescale_multiplier',
      'guidance',
      'width',
      'height',
      'vae_precision',
    ]);
  });

  it('gives LoRAs the weight and its slider bounds', () => {
    expect(fieldKeys(model('sdxl', 'lora'))).toEqual(['weight', 'weight_min', 'weight_max']);
  });

  it('adds fp8_storage alongside the preprocessor for a ControlNet', () => {
    expect(fieldKeys(model('sdxl', 'controlnet'))).toEqual(['preprocessor', 'fp8_storage']);
  });

  it('gives a ControlLoRA the preprocessor only', () => {
    // Not decided here any more: the backend's table has no `supported` row for a control LoRA, since
    // it is patched into a base model and the casting hooks would never fire.
    expect(fieldKeys(model('sdxl', 'control_lora'), false)).toEqual(['preprocessor']);
  });

  it('leaves the row out entirely when the backend says FP8 Storage does nothing for this model', () => {
    // The whole point of the change: a FLUX ControlNet, a GGUF main model or a model whose loader never
    // casts must not be offered a control that silently does nothing.
    expect(fieldKeys(model('flux', 'controlnet'), false)).toEqual(['preprocessor']);
    expect(fieldKeys(model('flux', 'main'), false)).not.toContain('fp8_storage');
  });
});

describe('getFieldsForModel scheduler row', () => {
  it("offers the injected policy's options and turns on at its default", () => {
    const field = schedulerField(model('anima', 'main'), ANIMA_POLICY);

    expect(field?.control).toEqual({
      kind: 'combobox',
      options: [
        { label: 'Euler', value: 'euler' },
        { label: 'ER-SDE', value: 'er_sde' },
      ],
    });
    expect(field?.defaultValue).toBe('euler');
  });

  it('leaves the row out where the graph ignores the scheduler and nothing is stored', () => {
    expect(fieldKeys(model('flux2', 'main'), false, IGNORED_POLICY)).not.toContain('scheduler');
  });

  it('keeps the row where the graph ignores the scheduler but a value is stored, so it can be cleared', () => {
    expect(fieldKeys(model('flux2', 'main', { scheduler: 'euler_a' }), false, IGNORED_POLICY)).toContain('scheduler');
  });

  it.each(['loading', 'unavailable'] as const)(
    'shows the row %s, keeping a stored value as its only choice',
    (rules) => {
      const pending = getFieldsForModel(model('flux2', 'main', { scheduler: 'euler_a' }), false, rules).find(
        (field) => field.key === 'scheduler'
      );

      expect(pending).toMatchObject({
        control: { kind: 'combobox', options: [{ label: 'euler_a', value: 'euler_a' }] },
        pending: rules,
      });
      expect(fieldKeys(model('flux2', 'main'), false, rules)).toContain('scheduler');
    }
  );

  it('keeps the row second, after the VAE', () => {
    expect(fieldKeys(model('sdxl', 'main'), false).slice(0, 3)).toEqual(['vae', 'scheduler', 'steps']);
  });
});

describe('validateDefaults', () => {
  const t = ((key: string) => key) as Parameters<typeof validateDefaults>[2];

  it('accepts the vae sentinel and a model key, rejects an empty string', () => {
    expect(validateDefaults(model('sdxl', 'main'), { vae: 'default' }, t)).toBeNull();
    expect(validateDefaults(model('sdxl', 'main'), { vae: 'some-model-key' }, t)).toBeNull();
    expect(validateDefaults(model('sdxl', 'main'), { vae: '' }, t)).not.toBeNull();
  });

  it('mirrors the backend LoRA bound validator: effective min must stay below max', () => {
    expect(validateDefaults(model('sdxl', 'lora'), { weight_max: 2, weight_min: -1 }, t)).toBeNull();
    expect(validateDefaults(model('sdxl', 'lora'), { weight_max: -1, weight_min: 1 }, t)).not.toBeNull();
    // Fallbacks are [-1, 2]: a min alone above 2 is invalid.
    expect(validateDefaults(model('sdxl', 'lora'), { weight_min: 3 }, t)).not.toBeNull();
  });

  it('requires an enabled weight to sit inside the effective range', () => {
    expect(validateDefaults(model('sdxl', 'lora'), { weight: 1.5 }, t)).toBeNull();
    expect(validateDefaults(model('sdxl', 'lora'), { weight: 2.5 }, t)).not.toBeNull();
    expect(validateDefaults(model('sdxl', 'lora'), { weight: 2.5, weight_max: 3 }, t)).toBeNull();
  });
});
