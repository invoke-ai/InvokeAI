import type { RootState } from 'app/store/store';
import type { Graph } from 'features/nodes/util/graph/Graph';
import { MetadataUtil } from 'features/nodes/util/graph/MetadataUtil';
import type { Invocation } from 'services/api/types';

import { SEAMLESS, VAE_LOADER } from './constants';

/**
 * Adds the seamless node to the graph and connects it to the model loader and denoise node.
 * Because the seamless node may insert a VAE loader node between the model loader and itself,
 * this function returns the terminal model loader node in the graph.
 * @param state The current Redux state
 * @param g The graph to add the seamless node to
 * @param denoise The denoise node in the graph
 * @param modelLoader The model loader node in the graph
 * @returns The terminal model loader node in the graph
 */
export const addSeamlessToGenerationTabGraph = (
  state: RootState,
  g: Graph,
  denoise: Invocation<'denoise_latents'>,
  modelLoader: Invocation<'main_model_loader'> | Invocation<'sdxl_model_loader'>
): Invocation<'main_model_loader'> | Invocation<'sdxl_model_loader'> | Invocation<'seamless'> => {
  const { seamlessXAxis, seamlessYAxis, vae } = state.generation;

  if (!seamlessXAxis && !seamlessYAxis) {
    return modelLoader;
  }

  const seamless = g.addNode({
    id: SEAMLESS,
    type: 'seamless',
    seamless_x: seamlessXAxis,
    seamless_y: seamlessYAxis,
  });

  const vaeLoader = vae
    ? g.addNode({
        type: 'vae_loader',
        id: VAE_LOADER,
        vae_model: vae,
      })
    : null;

  let terminalModelLoader: Invocation<'main_model_loader'> | Invocation<'sdxl_model_loader'> | Invocation<'seamless'> =
    modelLoader;

  if (seamlessXAxis) {
    MetadataUtil.add(g, {
      seamless_x: seamlessXAxis,
    });
    terminalModelLoader = seamless;
  }
  if (seamlessYAxis) {
    MetadataUtil.add(g, {
      seamless_y: seamlessYAxis,
    });
    terminalModelLoader = seamless;
  }

  // Seamless slots into the graph between the model loader and the denoise node
  g.deleteEdgesFrom(modelLoader, 'unet');
  g.deleteEdgesFrom(modelLoader, 'clip');

  g.addEdge(modelLoader, 'unet', seamless, 'unet');
  g.addEdge(vaeLoader ?? modelLoader, 'vae', seamless, 'unet');
  g.addEdge(seamless, 'unet', denoise, 'unet');

  return terminalModelLoader;
};
