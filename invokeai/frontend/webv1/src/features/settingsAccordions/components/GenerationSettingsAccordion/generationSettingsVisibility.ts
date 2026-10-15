import type { BaseModelType } from 'features/nodes/types/common';

const BASES_WITHOUT_STANDARD_SCHEDULER = new Set<BaseModelType>([
  'external',
  'flux',
  'flux2',
  'sd-3',
  'cogview4',
  'z-image',
  'qwen-image',
  'ernie-image',
  'anima',
  'krea-2',
  'wan',
  'ideogram-4',
  'minimax-h3',
]);

export const shouldShowStandardScheduler = (base: BaseModelType | null | undefined): boolean =>
  base === undefined || base === null || !BASES_WITHOUT_STANDARD_SCHEDULER.has(base);
