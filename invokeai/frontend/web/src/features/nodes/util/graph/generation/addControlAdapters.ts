import type {
  ControlAdapterEntity,
  ControlNetData,
  ImageWithDims,
  ProcessorConfig,
  T2IAdapterData,
} from 'features/controlLayers/store/types';
import type { ImageField } from 'features/nodes/types/common';
import { CONTROL_NET_COLLECT, T2I_ADAPTER_COLLECT } from 'features/nodes/util/graph/constants';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { BaseModelType, Invocation } from 'services/api/types';
import { assert } from 'tsafe';

export const addControlAdapters = (
  controlAdapters: ControlAdapterEntity[],
  g: Graph,
  denoise: Invocation<'denoise_latents'>,
  base: BaseModelType
): ControlAdapterEntity[] => {
  const validControlAdapters = controlAdapters.filter((ca) => isValidControlAdapter(ca, base));
  for (const ca of validControlAdapters) {
    if (ca.adapterType === 'controlnet') {
      addControlNetToGraph(ca, g, denoise);
    } else {
      addT2IAdapterToGraph(ca, g, denoise);
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

const addControlNetToGraph = (ca: ControlNetData, g: Graph, denoise: Invocation<'denoise_latents'>) => {
  const { id, beginEndStepPct, controlMode, imageObject, model, processedImageObject, processorConfig, weight } = ca;
  assert(model, 'ControlNet model is required');
  const controlImage = buildControlImage(
    imageObject?.image ?? null,
    processedImageObject?.image ?? null,
    processorConfig
  );
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
    image: controlImage,
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

const addT2IAdapterToGraph = (ca: T2IAdapterData, g: Graph, denoise: Invocation<'denoise_latents'>) => {
  const { id, beginEndStepPct, imageObject, model, processedImageObject, processorConfig, weight } = ca;
  assert(model, 'T2I Adapter model is required');
  const controlImage = buildControlImage(
    imageObject?.image ?? null,
    processedImageObject?.image ?? null,
    processorConfig
  );
  const t2iAdapterCollect = addT2IAdapterCollectorSafe(g, denoise);

  const t2iAdapter = g.addNode({
    id: `t2i_adapter_${id}`,
    type: 't2i_adapter',
    begin_step_percent: beginEndStepPct[0],
    end_step_percent: beginEndStepPct[1],
    resize_mode: 'just_resize',
    t2i_adapter_model: model,
    weight: weight,
    image: controlImage,
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

const isValidControlAdapter = (ca: ControlAdapterEntity, base: BaseModelType): boolean => {
  // Must be have a model that matches the current base and must have a control image
  const hasModel = Boolean(ca.model);
  const modelMatchesBase = ca.model?.base === base;
  const hasControlImage = Boolean(ca.imageObject || (ca.processedImageObject && ca.processorConfig));
  return hasModel && modelMatchesBase && hasControlImage;
};
