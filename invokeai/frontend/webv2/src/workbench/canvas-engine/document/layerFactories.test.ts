import { CONTROL_ADAPTER_DEFAULTS } from '@workbench/controlAdapters';
import { describe, expect, it } from 'vitest';

import { createControlLayer } from './layerFactories';

describe('createControlLayer (engine results and filters)', () => {
  it.each([
    ['anima', 'anima_lllite'],
    ['z-image', 'z_image_control'],
    ['sdxl', 'controlnet'],
  ] as const)('starts a %s layer on its kind with that kind’s defaults and the given model', (base, kind) => {
    expect(createControlLayer('Control Layer 1', 'c1', base, 'model-key').adapter).toEqual({
      ...CONTROL_ADAPTER_DEFAULTS[kind],
      model: 'model-key',
    });
  });
});
