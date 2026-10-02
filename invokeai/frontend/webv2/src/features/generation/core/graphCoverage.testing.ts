/** Settings fixtures that satisfy every base's required components, shared by graph and recall coverage. */

import type {
  ComponentPolicyContext,
  ComponentSlotPolicy,
  GenerateComponentValueKey,
  SupportedGenerateBase,
} from './baseGenerationPolicies';
import type { GenerateSettings, MainModelConfig, ModelIdentifierConfig } from './types';

import {
  getComponentSectionPolicy,
  getDefaultGenerateSettings,
  SUPPORTED_GENERATE_BASES,
} from './baseGenerationPolicies';
import { getIsPidSupportedBase } from './pid';

/** Compile both bundled and standalone component paths. */
export interface ModelShape {
  label: string;
  overrides: Partial<MainModelConfig>;
}

const DEFAULT_SHAPES: readonly ModelShape[] = [
  { label: 'diffusers', overrides: { format: 'diffusers' } },
  { label: 'standalone-components', overrides: { format: 'gguf_quantized' } },
];

/** Check per-base shape overrides against supported bases. */
export const SHAPE_OVERRIDES: Partial<Record<SupportedGenerateBase, readonly ModelShape[]>> = {
  // Ideogram's standalone fixture is a conditional checkpoint, not GGUF.
  'ideogram-4': [
    { label: 'diffusers', overrides: { format: 'diffusers' } },
    { label: 'standalone-components', overrides: { branch: 'conditional', format: 'checkpoint' } },
  ],
  // FLUX.2 dev and Klein need distinct encoder variants.
  flux2: [
    { label: 'dev-diffusers', overrides: { format: 'diffusers', variant: 'dev' } },
    { label: 'dev-standalone', overrides: { format: 'gguf_quantized', variant: 'dev' } },
    { label: 'klein-9b-standalone', overrides: { format: 'gguf_quantized', variant: 'klein_9b' } },
  ],
};

const shapesForBase = (base: SupportedGenerateBase): readonly ModelShape[] => SHAPE_OVERRIDES[base] ?? DEFAULT_SHAPES;

/** Search component candidates through policy rather than copy compatibility rules. */
const CANDIDATE_BASES = ['any', ...SUPPORTED_GENERATE_BASES] as const;
const CANDIDATE_VARIANTS = [
  undefined,
  'qwen3_06b',
  'qwen3_4b',
  'qwen3_8b',
  'large',
  'gigantic',
  'dev',
  'klein_4b',
  'klein_9b',
  'ministral3_3b',
  'qwen3_vl_4b',
  'qwen3_vl_8b',
] as const;
/** Ideogram 4's two transformer branches; every other main leaves the field unset. */
const CANDIDATE_BRANCHES = [undefined, 'conditional', 'unconditional'] as const;
/** VAE widths. A served row can constrain the width as well as the base -- Wan ships 16 and 48. */
const CANDIDATE_LATENT_CHANNELS = [undefined, 16, 48] as const;

export const candidatesForSlot = (slot: ComponentSlotPolicy): ModelIdentifierConfig[] => {
  const candidates: ModelIdentifierConfig[] = [];

  for (const type of slot.modelTypes) {
    for (const base of CANDIDATE_BASES) {
      for (const variant of CANDIDATE_VARIANTS) {
        for (const latentChannels of type === 'vae' ? CANDIDATE_LATENT_CHANNELS : [undefined]) {
          for (const branch of type === 'main' ? CANDIDATE_BRANCHES : [undefined]) {
            candidates.push({
              base,
              // Distinguish bundled component sources from single-file branch formats.
              format: slot.valueKind === 'main' ? (branch ? 'checkpoint' : 'diffusers') : undefined,
              key: `${base}-${type}-${variant ?? 'novariant'}${latentChannels ? `-${latentChannels}` : ''}${branch ? `-${branch}` : ''}`,
              name: `${base} ${type} ${variant ?? ''} ${branch ?? ''}`.trim(),
              type,
              variant: variant ?? null,
              ...(branch ? { branch } : {}),
              ...(latentChannels ? { latent_channels: latentChannels } : {}),
            });
          }
        }
      }
    }
  }

  return candidates;
};

export const buildContext = (
  model: MainModelConfig,
  settings: GenerateSettings,
  slots: readonly ComponentSlotPolicy[]
): ComponentPolicyContext => {
  // Derive keys from slots so new slots enter coverage automatically.
  const keys = new Set<GenerateComponentValueKey>(slots.map((slot) => slot.key));
  const selectedComponents = {} as ComponentPolicyContext['selectedComponents'];

  for (const key of keys) {
    selectedComponents[key] = settings[key] as never;
  }

  return { model, settings, selectedComponents };
};

/** Fill to a fixed point: selecting a component source can change required slots. */
const satisfyRequiredComponents = (
  model: MainModelConfig,
  initial: GenerateSettings
): { settings: GenerateSettings; filled: GenerateComponentValueKey[] } => {
  let settings = initial;
  const filled: GenerateComponentValueKey[] = [];

  for (let pass = 0; pass < 5; pass++) {
    const { slots } = getComponentSectionPolicy(model, settings);
    const context = buildContext(model, settings, slots);
    let changed = false;

    for (const slot of slots) {
      if (!slot.required?.(context) || settings[slot.key]) {
        continue;
      }

      const candidate = candidatesForSlot(slot).find((c) => !slot.filter || slot.filter(c, context));

      if (candidate) {
        settings = { ...settings, [slot.key]: candidate };
        filled.push(slot.key);
        changed = true;
      }
    }

    if (!changed) {
      break;
    }
  }

  return { filled: filled.sort(), settings };
};

const createModel = (base: SupportedGenerateBase, shape: ModelShape): MainModelConfig => ({
  base,
  key: `${base}-${shape.label}`,
  name: `${base} (${shape.label})`,
  type: 'main',
  ...shape.overrides,
});

export const satisfiedSettingsFor = (base: SupportedGenerateBase, shape: ModelShape) => {
  const model = createModel(base, shape);

  return {
    model,
    ...satisfyRequiredComponents(model, {
      ...getDefaultGenerateSettings(model),
      positivePrompt: 'a test prompt',
      seed: 1,
      seedMode: 'fixed',
    }),
  };
};

/** Every supported base in each of its component shapes. */
export const generateGraphCases = SUPPORTED_GENERATE_BASES.flatMap((base) =>
  shapesForBase(base).map((shape) => ({ base, label: `${base} / ${shape.label}`, shape }))
);

/** Required components plus every optional slot a compatible candidate fits, with PiD decoding on where supported. */
export const fullyFilledSettingsFor = (base: SupportedGenerateBase, shape: ModelShape) => {
  const { filled, model, settings: required } = satisfiedSettingsFor(base, shape);
  let settings: GenerateSettings = { ...required, pidMode: getIsPidSupportedBase(base) ? 'fit' : 'off' };
  const allFilled = new Set(filled);

  // Fill to a fixed point: a filled slot can reveal or constrain another.
  for (let pass = 0; pass < 5; pass++) {
    const { slots } = getComponentSectionPolicy(model, settings);
    const context = buildContext(model, settings, slots);
    let changed = false;

    for (const slot of slots) {
      if (settings[slot.key]) {
        continue;
      }

      const candidate = candidatesForSlot(slot).find((c) => !slot.filter || slot.filter(c, context));

      if (candidate) {
        settings = { ...settings, [slot.key]: candidate };
        allFilled.add(slot.key);
        changed = true;
      }
    }

    if (!changed) {
      break;
    }
  }

  return { filled: [...allFilled].sort(), model, settings };
};
