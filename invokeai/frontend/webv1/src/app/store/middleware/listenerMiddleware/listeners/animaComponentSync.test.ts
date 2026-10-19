import type { ModelIdentifierField } from 'features/nodes/types/common';
import { describe, expect, it } from 'vitest';

import { getAnimaComponentUpdates } from './animaComponentSync';

/**
 * Nothing validated `params.animaVaeModel` / `params.animaQwen3EncoderModel` when the model list
 * changed, so uninstalling a selected component left a dangling key in state. Both are required by
 * buildAnimaGraph, so the failure only surfaced at generation time (review 4998711432).
 */

const model = (key: string, base = 'anima'): ModelIdentifierField =>
  ({ key, hash: `${key}-hash`, name: key, base, type: 'vae' }) as ModelIdentifierField;

const animaVae = model('anima-vae');
const fluxVae = model('flux-vae', 'flux');
const encoder = model('qwen3-06b', 'any');

const base = {
  selectedVae: null,
  selectedEncoder: null,
  nativeVaes: [animaVae],
  compatibleVaes: [animaVae, fluxVae],
  availableEncoders: [encoder],
};

describe('getAnimaComponentUpdates', () => {
  it('leaves an available selection alone', () => {
    expect(getAnimaComponentUpdates({ ...base, selectedVae: animaVae, selectedEncoder: encoder })).toEqual({});
  });

  it('replaces a VAE that is no longer installed', () => {
    const updates = getAnimaComponentUpdates({
      ...base,
      selectedVae: model('uninstalled-vae'),
      selectedEncoder: encoder,
    });

    expect(updates).toEqual({ vae: animaVae });
  });

  it('clears a VAE that is no longer installed when nothing can replace it', () => {
    const updates = getAnimaComponentUpdates({
      ...base,
      selectedVae: model('uninstalled-vae'),
      selectedEncoder: encoder,
      nativeVaes: [],
      compatibleVaes: [],
    });

    expect(updates).toEqual({ vae: null });
  });

  it('replaces an encoder that is no longer installed', () => {
    const updates = getAnimaComponentUpdates({
      ...base,
      selectedVae: animaVae,
      selectedEncoder: model('uninstalled-encoder', 'any'),
    });

    expect(updates).toEqual({ encoder });
  });

  it('clears an encoder that is no longer installed when nothing can replace it', () => {
    const updates = getAnimaComponentUpdates({
      ...base,
      selectedVae: animaVae,
      selectedEncoder: model('uninstalled-encoder', 'any'),
      availableEncoders: [],
    });

    expect(updates).toEqual({ encoder: null });
  });

  // A FLUX VAE is a valid Anima selection, so it must survive a refetch rather than being swapped for
  // the native one on every model-list update.
  it('keeps a compatible fallback VAE that is still installed', () => {
    expect(getAnimaComponentUpdates({ ...base, selectedVae: fluxVae, selectedEncoder: encoder })).toEqual({});
  });

  // ...but when it has to pick, it picks the native one - same preference as the modelSelected listener.
  it('prefers a native VAE over a compatible fallback that sorts first', () => {
    const updates = getAnimaComponentUpdates({
      ...base,
      selectedVae: model('uninstalled-vae'),
      selectedEncoder: encoder,
      compatibleVaes: [fluxVae, animaVae],
    });

    expect(updates).toEqual({ vae: animaVae });
  });

  it('fills an empty slot when a component is available', () => {
    expect(getAnimaComponentUpdates(base)).toEqual({ vae: animaVae, encoder });
  });

  it('dispatches nothing when a slot is empty and nothing is installed', () => {
    expect(getAnimaComponentUpdates({ ...base, nativeVaes: [], compatibleVaes: [], availableEncoders: [] })).toEqual(
      {}
    );
  });
});
