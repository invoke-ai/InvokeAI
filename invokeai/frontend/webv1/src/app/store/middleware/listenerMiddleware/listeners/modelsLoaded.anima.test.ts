import type { AppDispatch, RootState } from 'app/store/store';
import { animaQwen3EncoderModelSelected, animaVaeModelSelected } from 'features/controlLayers/store/paramsSlice';
import type { Logger } from 'roarr';
import type { AnyModelConfig } from 'services/api/types';
import type { JsonObject } from 'type-fest';
import { describe, expect, it, vi } from 'vitest';

import { handleAnimaComponents } from './modelsLoaded';

/**
 * Wiring around `getAnimaComponentUpdates` (covered on its own in animaComponentSync.test.ts): the base
 * guard, and that a dangling selection is actually dispatched away. Both Anima component slots are
 * required by buildAnimaGraph, so a stale one only surfaced as a failed generation (review 4998711432).
 */

const animaVae = {
  key: 'anima-vae',
  hash: 'h',
  name: 'Wan 2.1 VAE',
  base: 'anima',
  type: 'vae',
  format: 'checkpoint',
} as unknown as AnyModelConfig;

const qwen3Encoder = {
  key: 'qwen3-06b',
  hash: 'h',
  name: 'Qwen3 0.6B',
  base: 'any',
  type: 'qwen3_encoder',
  variant: 'qwen3_06b',
  format: 'checkpoint',
} as unknown as AnyModelConfig;

const uninstalled = { key: 'gone', hash: 'h', name: 'Gone', base: 'anima', type: 'vae' };

const makeState = (params: Record<string, unknown>) =>
  ({
    params: {
      model: { key: 'anima-main', hash: 'h', name: 'Anima', base: 'anima', type: 'main' },
      animaVaeModel: null,
      animaQwen3EncoderModel: null,
      ...params,
    },
  }) as unknown as RootState;

const log = { debug: vi.fn() } as unknown as Logger<JsonObject>;

const run = (models: AnyModelConfig[], state: RootState) => {
  const dispatch = vi.fn() as unknown as AppDispatch;
  handleAnimaComponents(models, state, dispatch, log);
  return dispatch as unknown as ReturnType<typeof vi.fn>;
};

describe('handleAnimaComponents', () => {
  it('does nothing while another base is selected', () => {
    const state = makeState({
      model: { key: 'sdxl', hash: 'h', name: 'SDXL', base: 'sdxl', type: 'main' },
      animaVaeModel: uninstalled,
    });

    expect(run([], state)).not.toHaveBeenCalled();
  });

  it('clears a selected VAE that is no longer installed', () => {
    const dispatch = run(
      [qwen3Encoder],
      makeState({ animaVaeModel: uninstalled, animaQwen3EncoderModel: qwen3Encoder })
    );

    expect(dispatch).toHaveBeenCalledWith(animaVaeModelSelected(null));
  });

  it('replaces a selected VAE that is no longer installed when one is available', () => {
    const dispatch = run(
      [animaVae, qwen3Encoder],
      makeState({ animaVaeModel: uninstalled, animaQwen3EncoderModel: qwen3Encoder })
    );

    expect(dispatch).toHaveBeenCalledWith(animaVaeModelSelected(expect.objectContaining({ key: animaVae.key })));
  });

  it('fills both empty slots from the installed models', () => {
    const dispatch = run([animaVae, qwen3Encoder], makeState({}));

    expect(dispatch).toHaveBeenCalledWith(animaVaeModelSelected(expect.objectContaining({ key: animaVae.key })));
    expect(dispatch).toHaveBeenCalledWith(
      animaQwen3EncoderModelSelected(expect.objectContaining({ key: qwen3Encoder.key }))
    );
  });

  it('leaves an intact configuration alone', () => {
    const dispatch = run(
      [animaVae, qwen3Encoder],
      makeState({ animaVaeModel: animaVae, animaQwen3EncoderModel: qwen3Encoder })
    );

    expect(dispatch).not.toHaveBeenCalled();
  });
});
