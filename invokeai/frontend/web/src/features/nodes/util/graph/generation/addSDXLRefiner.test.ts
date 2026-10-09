import type { Invocation } from 'services/api/types';
import { describe, expect, it, vi } from 'vitest';

let nextId = 0;
vi.mock('features/controlLayers/konva/util', () => ({
  getPrefixedId: (prefix: string) => `${prefix}:${nextId++}`,
}));

const refinerModel = {
  key: 'refiner-model',
  hash: 'refiner-hash',
  name: 'SDXL Refiner',
  base: 'sdxl',
  type: 'main',
};

vi.mock('features/metadata/util/modelFetchingHelpers', () => ({
  fetchModelConfigWithTypeGuard: vi.fn(() => Promise.resolve(refinerModel)),
}));

import { addSDXLRefiner } from './addSDXLRefiner';
import { Graph } from './Graph';

describe('addSDXLRefiner', () => {
  it('does not apply HiDiffusion to the unsupported refiner stage', async () => {
    const g = new Graph('test');
    const denoise = g.addNode({ type: 'denoise_latents', id: 'base-denoise' } as Invocation<'denoise_latents'>);
    const posCond = g.addNode({ type: 'sdxl_compel_prompt', id: 'pos' } as Invocation<'sdxl_compel_prompt'>);
    const negCond = g.addNode({ type: 'sdxl_compel_prompt', id: 'neg' } as Invocation<'sdxl_compel_prompt'>);
    const l2i = g.addNode({ type: 'l2i', id: 'l2i' } as Invocation<'l2i'>);

    const state = {
      params: {
        refinerModel,
        refinerPositiveAestheticScore: 6,
        refinerNegativeAestheticScore: 2.5,
        refinerSteps: 20,
        refinerScheduler: 'euler',
        refinerCFGScale: 7.5,
        refinerStart: 0.8,
        hiDiffusionEnabled: true,
        hiDiffusionRauNetEnabled: true,
        hiDiffusionWindowAttnEnabled: true,
        hiDiffusionT1Ratio: 0.4,
        hiDiffusionT2Ratio: 0.3,
      },
    } as never;

    await addSDXLRefiner(state, g, denoise, null, posCond, negCond, l2i);

    const refinerDenoise = Object.values(g.getGraph().nodes).find(
      (node) => node.type === 'denoise_latents' && node.id !== denoise.id
    ) as Invocation<'denoise_latents'> | undefined;
    expect(refinerDenoise).toBeDefined();
    expect(refinerDenoise?.hidiffusion).toBeUndefined();
    expect(refinerDenoise?.hidiffusion_raunet).toBeUndefined();
    expect(refinerDenoise?.hidiffusion_window_attn).toBeUndefined();
    expect(refinerDenoise?.hidiffusion_t1_ratio).toBeUndefined();
    expect(refinerDenoise?.hidiffusion_t2_ratio).toBeUndefined();
  });
});
