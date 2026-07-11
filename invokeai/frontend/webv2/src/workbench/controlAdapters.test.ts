import { describe, expect, it } from 'vitest';

import { CONTROL_ADAPTER_DEFAULTS, normalizeControlAdapter } from './controlAdapters';

describe('normalizeControlAdapter', () => {
  it.each([
    ['controlnet', -1, [0, 0.75]],
    ['t2i_adapter', 2, [0.2, 0.8]],
    ['control_lora', -1, [0, 1]],
    ['z_image_control', 0, [0.1, 0.9]],
  ] as const)('preserves valid %s numeric values', (kind, weight, beginEndStepPct) => {
    const adapter = { ...CONTROL_ADAPTER_DEFAULTS[kind], beginEndStepPct, weight };
    expect(normalizeControlAdapter(adapter)).toEqual(adapter);
  });

  it.each([
    ['controlnet', Number.NaN],
    ['controlnet', Number.POSITIVE_INFINITY],
    ['controlnet', -1.01],
    ['controlnet', 2.01],
    ['z_image_control', -0.01],
    ['z_image_control', 2.01],
  ] as const)('repairs invalid %s weight %s with its default', (kind, weight) => {
    const adapter = { ...CONTROL_ADAPTER_DEFAULTS[kind], weight };
    expect(normalizeControlAdapter(adapter)).toMatchObject({ kind, weight: CONTROL_ADAPTER_DEFAULTS[kind].weight });
  });

  it.each([
    [Number.NaN, 1],
    [0, Number.POSITIVE_INFINITY],
    [-0.01, 1],
    [0, 1.01],
    [0.5, 0.5],
    [0.8, 0.2],
  ])('repairs invalid step range [%s, %s] with adapter defaults', (begin, end) => {
    const adapter = { ...CONTROL_ADAPTER_DEFAULTS.z_image_control, beginEndStepPct: [begin, end] };
    expect(normalizeControlAdapter(adapter)).toMatchObject({ beginEndStepPct: [0, 1] });
  });
});
