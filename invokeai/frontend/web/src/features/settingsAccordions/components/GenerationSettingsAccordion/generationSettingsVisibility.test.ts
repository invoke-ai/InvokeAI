import { describe, expect, it } from 'vitest';

import { shouldShowStandardScheduler } from './generationSettingsVisibility';

describe('shouldShowStandardScheduler', () => {
  it.each(['sd-1', 'sd-2', 'sdxl', 'sdxl-refiner', undefined] as const)(
    'shows the standard scheduler for %s',
    (base) => {
      expect(shouldShowStandardScheduler(base)).toBe(true);
    }
  );

  it.each(['external', 'flux', 'flux2', 'sd-3', 'cogview4', 'z-image', 'qwen-image', 'anima', 'krea-2'] as const)(
    'hides the standard scheduler for %s',
    (base) => {
      expect(shouldShowStandardScheduler(base)).toBe(false);
    }
  );
});
