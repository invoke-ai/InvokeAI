import type { RootState } from 'app/store/store';
import { krea2Qwen3VlEncoderModelSelected, krea2VaeModelSelected } from 'features/controlLayers/store/paramsSlice';
import type { AnyModelConfig } from 'services/api/types';
import { describe, expect, it, vi } from 'vitest';

import { handleKrea2Components } from './modelsLoaded';

const mainModel = {
  key: 'krea-main',
  hash: 'main-hash',
  name: 'Krea Main',
  base: 'krea-2',
  type: 'main',
  format: 'gguf_quantized',
} as const;
const vae = {
  key: 'vae',
  hash: 'vae-hash',
  name: 'Qwen Image VAE',
  base: 'qwen-image',
  type: 'vae',
  format: 'checkpoint',
} as const;
const encoder = {
  key: 'encoder',
  hash: 'encoder-hash',
  name: 'Qwen3-VL Encoder',
  base: 'any',
  type: 'qwen3_vl_encoder',
  format: 'qwen3_vl_encoder',
} as const;

const makeState = (overrides: Record<string, unknown> = {}) =>
  ({
    params: {
      model: mainModel,
      krea2VaeModel: null,
      krea2Qwen3VlEncoderModel: null,
      ...overrides,
    },
  }) as unknown as RootState;

describe('handleKrea2Components', () => {
  it('selects standalone components when a deferred non-Diffusers model arrives in the fulfilled list', () => {
    const dispatch = vi.fn();

    handleKrea2Components(
      [mainModel, vae, encoder] as unknown as AnyModelConfig[],
      makeState(),
      dispatch,
      null as never
    );

    expect(dispatch).toHaveBeenCalledWith(krea2VaeModelSelected(expect.objectContaining({ key: vae.key })));
    expect(dispatch).toHaveBeenCalledWith(
      krea2Qwen3VlEncoderModelSelected(expect.objectContaining({ key: encoder.key }))
    );
  });

  it('clears stale standalone components when a deferred Diffusers model arrives in the fulfilled list', () => {
    const dispatch = vi.fn();
    const diffusersMain = { ...mainModel, format: 'diffusers' } as const;
    const state = makeState({ model: diffusersMain, krea2VaeModel: vae, krea2Qwen3VlEncoderModel: encoder });

    handleKrea2Components([diffusersMain, vae, encoder] as unknown as AnyModelConfig[], state, dispatch, null as never);

    expect(dispatch).toHaveBeenCalledWith(krea2VaeModelSelected(null));
    expect(dispatch).toHaveBeenCalledWith(krea2Qwen3VlEncoderModelSelected(null));
  });
});
