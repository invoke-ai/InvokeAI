import { z } from 'zod';

/**
 * Zod schemas backing every model-manager form. All user input is validated
 * through these before touching the API, so error states are consistent and
 * the API layer can assume well-formed values.
 */

const trimmed = z.string().trim();

/** URL, HuggingFace repo id, or absolute local path. */
export const installSourceSchema = z.object({
  accessToken: trimmed.optional(),
  inplace: z.boolean(),
  source: trimmed
    .min(1, 'Enter a URL, HuggingFace repo id, or local path.')
    .refine(
      (value) => !value.includes(' ') || value.startsWith('/') || /^[A-Za-z]:[\\/]/.test(value),
      'URLs and repo ids cannot contain spaces.'
    ),
});

export type InstallSourceFormValues = z.infer<typeof installSourceSchema>;

/** `owner/name` with optional `:variant[:path]` qualifiers. */
export const huggingFaceRepoSchema = z.object({
  repo: trimmed
    .min(1, 'Enter a HuggingFace repo id.')
    .regex(/^[\w.-]+\/[\w.-]+(:[\w./-]*)*$/, 'Expected an owner/repo id, e.g. black-forest-labs/FLUX.1-dev.'),
});

export type HuggingFaceRepoFormValues = z.infer<typeof huggingFaceRepoSchema>;

export const scanFolderSchema = z.object({
  path: trimmed.min(1, 'Enter an absolute folder path on the server.'),
});

export type ScanFolderFormValues = z.infer<typeof scanFolderSchema>;

export const hfTokenSchema = z.object({
  token: trimmed.min(1, 'Paste a HuggingFace access token.'),
});

export type HFTokenFormValues = z.infer<typeof hfTokenSchema>;

export const civitaiKeySchema = z.object({
  key: trimmed.min(1, 'Paste a Civitai API key.'),
});

export type CivitaiKeyFormValues = z.infer<typeof civitaiKeySchema>;

export const modelEditSchema = z.object({
  base: trimmed.min(1, 'Base architecture is required.'),
  description: trimmed.max(2000, 'Keep the description under 2000 characters.'),
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
  vaePrecision: z.enum(['fp16', 'fp32']).nullable(),
  width: optionalBoundedNumber(64, 8192, 'Width').refine(
    (value) => value === null || value % 8 === 0,
    'Width must be a multiple of 8.'
  ),
});

export type MainDefaultSettingsFormValues = z.infer<typeof mainDefaultSettingsSchema>;

export const loraDefaultSettingsSchema = z.object({
  weight: optionalBoundedNumber(-10, 10, 'Weight'),
});

export type LoraDefaultSettingsFormValues = z.infer<typeof loraDefaultSettingsSchema>;

export const triggerPhraseSchema = trimmed
  .min(1, 'Trigger phrase cannot be empty.')
  .max(200, 'Keep trigger phrases under 200 characters.');
