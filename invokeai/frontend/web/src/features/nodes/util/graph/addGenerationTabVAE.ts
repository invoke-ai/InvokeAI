import type { RootState } from 'app/store/store';
import type { Graph } from 'features/nodes/util/graph/Graph';
import { MetadataUtil } from 'features/nodes/util/graph/MetadataUtil';
import type { Invocation } from 'services/api/types';

import { VAE_LOADER } from './constants';

export const addGenerationTabVAE = (
  state: RootState,
  g: Graph,
  modelLoader: Invocation<'main_model_loader'> | Invocation<'sdxl_model_loader'>,
  l2i: Invocation<'l2i'>,
  i2l: Invocation<'i2l'> | null,
  seamless: Invocation<'seamless'> | null
): void => {
  const { vae } = state.generation;

  // The seamless helper also adds the VAE loader... so we need to check if it's already there
  const shouldAddVAELoader = !g.hasNode(VAE_LOADER) && vae;
  const vaeLoader = shouldAddVAELoader
    ? g.addNode({
        type: 'vae_loader',
        id: VAE_LOADER,
        vae_model: vae,
      })
    : null;

  const vaeSource = seamless ? seamless : vaeLoader ? vaeLoader : modelLoader;
  g.addEdge(vaeSource, 'vae', l2i, 'vae');
  if (i2l) {
    g.addEdge(vaeSource, 'vae', i2l, 'vae');
  }

  if (vae) {
    MetadataUtil.add(g, { vae });
  }
};
