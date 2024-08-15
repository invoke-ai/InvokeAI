import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type {
  CanvasControlLayerState,
  ControlNetConfig,
  Rect,
  T2IAdapterConfig,
} from 'features/controlLayers/store/types';
import { CONTROL_NET_COLLECT, T2I_ADAPTER_COLLECT } from 'features/nodes/util/graph/constants';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { BaseModelType, ImageDTO, Invocation } from 'services/api/types';
import { assert } from 'tsafe';

export const addControlAdapters = async (
  manager: CanvasManager,
  layers: CanvasControlLayerState[],
  g: Graph,
  bbox: Rect,
  denoise: Invocation<'denoise_latents'>,
  base: BaseModelType
): Promise<CanvasControlLayerState[]> => {
  const validControlLayers = layers
    .filter((layer) => layer.isEnabled)
    .filter((layer) => isValidControlAdapter(layer.controlAdapter, base));

  for (const layer of validControlLayers) {
    const adapter = manager.controlLayerAdapters.get(layer.id);
    assert(adapter, 'Adapter not found');
    const imageDTO = await adapter.renderer.rasterize(bbox);
    if (layer.controlAdapter.type === 'controlnet') {
      await addControlNetToGraph(g, layer, imageDTO, denoise);
    } else {
      await addT2IAdapterToGraph(g, layer, imageDTO, denoise);
    }
  }
  return validControlLayers;
};

const addControlNetCollectorSafe = (g: Graph, denoise: Invocation<'denoise_latents'>): Invocation<'collect'> => {
  try {
    // Attempt to retrieve the collector
    const controlNetCollect = g.getNode(CONTROL_NET_COLLECT);
    assert(controlNetCollect.type === 'collect');
    return controlNetCollect;
  } catch {
    // Add the ControlNet collector
    const controlNetCollect = g.addNode({
      id: CONTROL_NET_COLLECT,
      type: 'collect',
    });
    g.addEdge(controlNetCollect, 'collection', denoise, 'control');
    return controlNetCollect;
  }
};

const addControlNetToGraph = (
  g: Graph,
  layer: CanvasControlLayerState,
  imageDTO: ImageDTO,
  denoise: Invocation<'denoise_latents'>
) => {
  const { id, controlAdapter } = layer;
  assert(controlAdapter.type === 'controlnet');
  const { beginEndStepPct, model, weight, controlMode } = controlAdapter;
  assert(model !== null);
  const { image_name } = imageDTO;

  const controlNetCollect = addControlNetCollectorSafe(g, denoise);

  const controlNet = g.addNode({
    id: `control_net_${id}`,
    type: 'controlnet',
    begin_step_percent: beginEndStepPct[0],
    end_step_percent: beginEndStepPct[1],
    control_mode: controlMode,
    resize_mode: 'just_resize',
    control_model: model,
    control_weight: weight,
    image: { image_name },
  });
  g.addEdge(controlNet, 'control', controlNetCollect, 'item');
};

const addT2IAdapterCollectorSafe = (g: Graph, denoise: Invocation<'denoise_latents'>): Invocation<'collect'> => {
  try {
    // You see, we've already got one!
    const t2iAdapterCollect = g.getNode(T2I_ADAPTER_COLLECT);
    assert(t2iAdapterCollect.type === 'collect');
    return t2iAdapterCollect;
  } catch {
    const t2iAdapterCollect = g.addNode({
      id: T2I_ADAPTER_COLLECT,
      type: 'collect',
    });

    g.addEdge(t2iAdapterCollect, 'collection', denoise, 't2i_adapter');

    return t2iAdapterCollect;
  }
};

const addT2IAdapterToGraph = (
  g: Graph,
  layer: CanvasControlLayerState,
  imageDTO: ImageDTO,
  denoise: Invocation<'denoise_latents'>
) => {
  const { id, controlAdapter } = layer;
  assert(controlAdapter.type === 't2i_adapter');
  const { beginEndStepPct, model, weight } = controlAdapter;
  assert(model !== null);
  const { image_name } = imageDTO;

  const t2iAdapterCollect = addT2IAdapterCollectorSafe(g, denoise);

  const t2iAdapter = g.addNode({
    id: `t2i_adapter_${id}`,
    type: 't2i_adapter',
    begin_step_percent: beginEndStepPct[0],
    end_step_percent: beginEndStepPct[1],
    resize_mode: 'just_resize',
    t2i_adapter_model: model,
    weight: weight,
    image: { image_name },
  });

  g.addEdge(t2iAdapter, 't2i_adapter', t2iAdapterCollect, 'item');
};

const isValidControlAdapter = (controlAdapter: ControlNetConfig | T2IAdapterConfig, base: BaseModelType): boolean => {
  // Must be have a model
  const hasModel = Boolean(controlAdapter.model);
  // Model must match the current base model
  const modelMatchesBase = controlAdapter.model?.base === base;
  return hasModel && modelMatchesBase;
};
