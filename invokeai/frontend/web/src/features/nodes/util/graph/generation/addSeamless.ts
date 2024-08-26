import type { RootState } from 'app/store/store';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { Invocation } from 'services/api/types';

/**
 * Adds the seamless node to the graph and connects it to the model loader and denoise node.
 * Because the seamless node may insert a VAE loader node between the model loader and itself,
 * future nodes should be connected to the return value of this function.
 * @param state The current Redux state
 * @param g The graph to add the seamless node to
 * @param denoise The denoise node in the graph
 * @param modelLoader The model loader node in the graph
 * @param vaeLoader The VAE loader node in the graph, if it exists
 * @returns The seamless node, if it was added to the graph
 */
export const addSeamless = (
  state: RootState,
  g: Graph,
  denoise: Invocation<'denoise_latents'>,
  modelLoader: Invocation<'main_model_loader'> | Invocation<'sdxl_model_loader'>,
  vaeLoader: Invocation<'vae_loader'> | null
): Invocation<'seamless'> | null => {
  const { seamlessXAxis: seamless_x, seamlessYAxis: seamless_y } = state.params;

  if (!seamless_x && !seamless_y) {
    return null;
  }

  const seamless = g.addNode({
    type: 'seamless',
    id: getPrefixedId('seamless'),
    seamless_x,
    seamless_y,
  });

  g.upsertMetadata({
    seamless_x: seamless_x || undefined,
    seamless_y: seamless_y || undefined,
  });

  // Seamless slots into the graph between the model loader and the denoise node
  g.deleteEdgesFrom(modelLoader, ['unet']);
  g.deleteEdgesFrom(modelLoader, ['vae']);

  g.addEdge(modelLoader, 'unet', seamless, 'unet');
  g.addEdge(vaeLoader ?? modelLoader, 'vae', seamless, 'vae');
  g.addEdge(seamless, 'unet', denoise, 'unet');

  return seamless;
};
