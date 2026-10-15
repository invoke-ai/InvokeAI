/** Frontend graph-builder support is separate from backend architecture awareness. */

import type { KnownGenerationModelBase } from '@features/generation/core/contracts';

export const SUPPORTED_GENERATE_BASES = [
  'sd-1',
  'sd-2',
  'sdxl',
  'sd-3',
  'flux',
  'flux2',
  'cogview4',
  'ernie-image',
  'qwen-image',
  'z-image',
  'ideogram-4',
  'krea-2',
  'wan',
  'anima',
] as const satisfies readonly KnownGenerationModelBase[];

export type SupportedGenerateBase = (typeof SUPPORTED_GENERATE_BASES)[number];

const SUPPORTED = new Set<string>(SUPPORTED_GENERATE_BASES);

export const isSupportedGenerateBase = (base: string): base is SupportedGenerateBase => SUPPORTED.has(base);
