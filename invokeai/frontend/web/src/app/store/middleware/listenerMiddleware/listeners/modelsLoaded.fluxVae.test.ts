import type { AppDispatch, RootState } from 'app/store/store';
import { fluxVAESelected } from 'features/controlLayers/store/paramsSlice';
import type { Logger } from 'roarr';
import type { AnyModelConfig } from 'services/api/types';
import type { JsonObject } from 'type-fest';
import { describe, expect, it, vi } from 'vitest';

import { handleFLUXVAEModels } from './modelsLoaded';

/**
 * `params.fluxVAE` is a FLUX.1-only slot: buildFLUXGraph feeds it to `flux_model_loader.vae_model` in the
 * FLUX.1 branch (FLUX.2 uses `params.flux2VaeModel`), and its picker is built from
 * `isFlux1VAEModelConfig`. Defaulting it from the wider flux+flux2 pool silently put a FLUX.2 VAE into a
 * slot the user could not see it in and that FLUX.1 cannot load - and, downstream, produced a
 * `metadata.vae` value that no recall handler would accept, so the VAE row vanished from the panel.
 */

const flux1Vae = {
  key: 'flux1-vae',
  hash: 'h',
  name: 'FLUX.1 VAE',
  base: 'flux',
  type: 'vae',
  format: 'checkpoint',
} as unknown as AnyModelConfig;

const flux2Vae = {
  ...flux1Vae,
  key: 'flux2-vae',
  name: 'FLUX.2 VAE',
  base: 'flux2',
} as unknown as AnyModelConfig;

const flux1MainWithVaeSubmodel = {
  key: 'flux1-main',
  hash: 'h',
  name: 'FLUX.1 dev (diffusers)',
  base: 'flux',
  type: 'main',
  format: 'diffusers',
  submodels: { vae: {} },
} as unknown as AnyModelConfig;

const makeState = (fluxVAE: unknown = null) => ({ params: { fluxVAE } }) as unknown as RootState;

const log = { debug: vi.fn() } as unknown as Logger<JsonObject>;

const run = (models: AnyModelConfig[], fluxVAE: unknown = null) => {
  const dispatch = vi.fn() as unknown as AppDispatch;
  handleFLUXVAEModels(models, makeState(fluxVAE), dispatch, log);
  return dispatch as unknown as ReturnType<typeof vi.fn>;
};

describe('handleFLUXVAEModels', () => {
  it('selects a FLUX.1 VAE', () => {
    const dispatch = run([flux1Vae]);

    expect(dispatch).toHaveBeenCalledWith(fluxVAESelected(expect.objectContaining({ key: flux1Vae.key })));
  });

  it('never selects a FLUX.2 VAE, even when it is the only VAE installed', () => {
    const dispatch = run([flux2Vae]);

    expect(dispatch).not.toHaveBeenCalled();
  });

  it('skips a FLUX.2 VAE that sorts before the FLUX.1 one', () => {
    const dispatch = run([flux2Vae, flux1Vae]);

    expect(dispatch).toHaveBeenCalledTimes(1);
    expect(dispatch).toHaveBeenCalledWith(fluxVAESelected(expect.objectContaining({ key: flux1Vae.key })));
  });

  // The picker offers bundled VAE submodels, so the auto-default may pick one too - and the recall
  // handler now accepts it rather than dropping the row.
  it('accepts a FLUX.1 main model bundled VAE submodel', () => {
    const dispatch = run([flux1MainWithVaeSubmodel]);

    expect(dispatch).toHaveBeenCalledWith(
      fluxVAESelected(expect.objectContaining({ key: flux1MainWithVaeSubmodel.key }))
    );
  });

  it('clears a selection that is no longer available', () => {
    const dispatch = run([], flux1Vae);

    expect(dispatch).toHaveBeenCalledWith(fluxVAESelected(null));
  });

  it('leaves an available selection alone', () => {
    const dispatch = run([flux1Vae], flux1Vae);

    expect(dispatch).not.toHaveBeenCalled();
  });
});
