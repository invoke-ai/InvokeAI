import { z } from 'zod';

/**
 * Zod schemas for the model-manager forms with structured validation (edit,
 * default settings, trigger phrases). User input goes through these before
 * touching the API, so error states are consistent and the API layer can
 * assume well-formed values.
 */

const trimmed = z.string().trim();

export const modelEditSchema = z.object({
  base: trimmed.min(1, 'Base architecture is required.'),
  configPath: trimmed,
  description: trimmed.max(2000, 'Keep the description under 2000 characters.'),
  format: trimmed.min(1, 'Format is required.'),
  name: trimmed.min(1, 'Name is required.').max(200, 'Keep the name under 200 characters.'),
  predictionType: z.union([z.literal(''), z.enum(['epsilon', 'v_prediction', 'sample'])]),
  sourceUrl: z.union([z.literal(''), z.url({ error: 'Must be a valid http(s) URL.' })]),
  type: trimmed.min(1, 'Model type is required.'),
  variant: trimmed,
});

export type ModelEditFormValues = z.infer<typeof modelEditSchema>;

const optionalBoundedNumber = (min: number, max: number, label: string) =>
  z
    .number({ message: `${label} must be a number.` })
    .min(min, `${label} must be at least ${min}.`)
    .max(max, `${label} must be at most ${max}.`)
    .nullable();

/**
 * Per-model generation defaults. Every field is nullable — null means "no
 * default, use the app-level setting" and maps to an unchecked toggle.
 */
export const mainDefaultSettingsSchema = z.object({
  cfgRescaleMultiplier: optionalBoundedNumber(0, 0.99, 'CFG rescale multiplier'),
  cfgScale: optionalBoundedNumber(1, 200, 'CFG scale'),
  guidance: optionalBoundedNumber(1, 20, 'Guidance'),
  height: optionalBoundedNumber(64, 8192, 'Height').refine(
    (value) => value === null || value % 8 === 0,
    'Height must be a multiple of 8.'
  ),
  scheduler: trimmed.nullable(),
  steps: optionalBoundedNumber(1, 10000, 'Steps'),
  // A VAE model key, or the backend sentinel 'default' for "the compatible
  // default VAE"; null inherits the app-level setting.
  vae: trimmed.min(1, 'Choose a VAE or turn the field off.').nullable(),
  vaePrecision: z.enum(['fp16', 'fp32']).nullable(),
  width: optionalBoundedNumber(64, 8192, 'Width').refine(
    (value) => value === null || value % 8 === 0,
    'Width must be a multiple of 8.'
  ),
});

export type MainDefaultSettingsFormValues = z.infer<typeof mainDefaultSettingsSchema>;

// Mirrors the backend validator (configs/lora.py): the effective slider range
// falls back to [-1, 2], min must stay below max, and an enabled weight must
// sit inside the effective range — otherwise the save 409s.
export const loraDefaultSettingsSchema = z
  .object({
    weight: optionalBoundedNumber(-10, 10, 'Weight'),
    weightMax: optionalBoundedNumber(-10, 10, 'Weight max'),
    weightMin: optionalBoundedNumber(-10, 10, 'Weight min'),
  })
  .superRefine((values, context) => {
    const effectiveMin = values.weightMin ?? -1;
    const effectiveMax = values.weightMax ?? 2;

    if (effectiveMin >= effectiveMax) {
      context.addIssue({
        code: 'custom',
        message: `Weight min (${effectiveMin}) must be less than weight max (${effectiveMax}).`,
        path: ['weightMin'],
      });

      return;
    }

    if (values.weight !== null && (values.weight < effectiveMin || values.weight > effectiveMax)) {
      context.addIssue({
        code: 'custom',
        message: `Weight (${values.weight}) must be within [${effectiveMin}, ${effectiveMax}].`,
        path: ['weight'],
      });
    }
  });

export type LoraDefaultSettingsFormValues = z.infer<typeof loraDefaultSettingsSchema>;

/** POSIX (/x), Windows drive (C:\ or C:/), or UNC (\\server) — the shapes in-place installs record. */
export const isAbsoluteModelPath = (path: string): boolean =>
  path.startsWith('/') || path.startsWith('\\\\') || /^[A-Za-z]:[\\/]/.test(path);

/**
 * Mirrors the backend's `models_path / model.path` resolution: managed models
 * store paths relative to the models directory, in-place installs store them
 * absolute. Falls back to the stored path when the directory is unknown.
 */
export const resolveModelAbsolutePath = (path: string, modelsDir: string | null): string =>
  isAbsoluteModelPath(path) || !modelsDir ? path : `${modelsDir.replace(/\/+$/, '')}/${path}`;

/**
 * Replacement path for a relocated model. Only absolute paths are accepted:
 * managed models store paths relative to the models directory and must not be
 * relocated this way.
 */
export const modelPathSchema = trimmed
  .min(1, 'Path is required.')
  .refine(isAbsoluteModelPath, 'Must be an absolute path.');

export const triggerPhraseSchema = trimmed
  .min(1, 'Trigger phrase cannot be empty.')
  .max(200, 'Keep trigger phrases under 200 characters.');
