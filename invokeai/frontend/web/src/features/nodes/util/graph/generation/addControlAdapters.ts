import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type {
  CanvasControlAdapterState,
  CanvasControlNetState,
  ImageWithDims,
  ProcessorConfig,
  Rect,
  CanvasT2IAdapterState,
} from 'features/controlLayers/store/types';
import type { ImageField } from 'features/nodes/types/common';
import { CONTROL_NET_COLLECT, T2I_ADAPTER_COLLECT } from 'features/nodes/util/graph/constants';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { BaseModelType, Invocation } from 'services/api/types';
import { assert } from 'tsafe';

export const addControlAdapters = async (
  manager: CanvasManager,
  controlAdapters: CanvasControlAdapterState[],
  g: Graph,
  bbox: Rect,
  denoise: Invocation<'denoise_latents'>,
  base: BaseModelType
): Promise<CanvasControlAdapterState[]> => {
  const validControlAdapters = controlAdapters.filter((ca) => isValidControlAdapter(ca, base));
  for (const ca of validControlAdapters) {
    if (ca.adapterType === 'controlnet') {
      await addControlNetToGraph(manager, ca, g, bbox, denoise);
    } else {
      await addT2IAdapterToGraph(manager, ca, g, bbox, denoise);
    }
  }
  return validControlAdapters;
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

const addControlNetToGraph = async (
  manager: CanvasManager,
  ca: CanvasControlNetState,
  g: Graph,
  bbox: Rect,
  denoise: Invocation<'denoise_latents'>
) => {
  const { id, beginEndStepPct, controlMode, model, weight } = ca;
  assert(model, 'ControlNet model is required');
  const { image_name } = await manager.getControlAdapterImage({ id: ca.id, bbox, preview: true });

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

const addT2IAdapterToGraph = async (
  manager: CanvasManager,
  ca: CanvasT2IAdapterState,
  g: Graph,
  bbox: Rect,
  denoise: Invocation<'denoise_latents'>
) => {
  const { id, beginEndStepPct, model, weight } = ca;
  assert(model, 'T2I Adapter model is required');
  const { image_name } = await manager.getControlAdapterImage({ id: ca.id, bbox, preview: true });

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

const buildControlImage = (
  image: ImageWithDims | null,
  processedImage: ImageWithDims | null,
  processorConfig: ProcessorConfig | null
): ImageField => {
  if (processedImage && processorConfig) {
    // We've processed the image in the app - use it for the control image.
    return {
      image_name: processedImage.name,
    };
  } else if (image) {
    // No processor selected, and we have an image - the user provided a processed image, use it for the control image.
    return {
      image_name: image.name,
    };
  }
  assert(false, 'Attempted to add unprocessed control image');
};

const isValidControlAdapter = (ca: CanvasControlAdapterState, base: BaseModelType): boolean => {
  // Must be have a model that matches the current base and must have a control image
  const hasModel = Boolean(ca.model);
  const modelMatchesBase = ca.model?.base === base;
  const hasControlImage = Boolean(ca.imageObject || (ca.processedImageObject && ca.processorConfig));
  return hasModel && modelMatchesBase && hasControlImage;
};
