import type { CanvasIPAdapterState } from 'features/controlLayers/store/types';
import { IP_ADAPTER_COLLECT } from 'features/nodes/util/graph/constants';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { BaseModelType, Invocation } from 'services/api/types';
import { assert } from 'tsafe';

export const addIPAdapters = (
  ipAdapters: CanvasIPAdapterState[],
  g: Graph,
  denoise: Invocation<'denoise_latents'>,
  base: BaseModelType
): CanvasIPAdapterState[] => {
  const validIPAdapters = ipAdapters.filter((ipa) => isValidIPAdapter(ipa, base));
  for (const ipa of validIPAdapters) {
    addIPAdapter(ipa, g, denoise);
  }
  return validIPAdapters;
};

export const addIPAdapterCollectorSafe = (g: Graph, denoise: Invocation<'denoise_latents'>): Invocation<'collect'> => {
  try {
    // You see, we've already got one!
    const ipAdapterCollect = g.getNode(IP_ADAPTER_COLLECT);
    assert(ipAdapterCollect.type === 'collect');
    return ipAdapterCollect;
  } catch {
    const ipAdapterCollect = g.addNode({
      id: IP_ADAPTER_COLLECT,
      type: 'collect',
    });
    g.addEdge(ipAdapterCollect, 'collection', denoise, 'ip_adapter');
    return ipAdapterCollect;
  }
};

const addIPAdapter = (ipa: CanvasIPAdapterState, g: Graph, denoise: Invocation<'denoise_latents'>) => {
  const { id, weight, model, clipVisionModel, method, beginEndStepPct, imageObject } = ipa;
  assert(imageObject, 'IP Adapter image is required');
  assert(model, 'IP Adapter model is required');
  const ipAdapterCollect = addIPAdapterCollectorSafe(g, denoise);

  const ipAdapter = g.addNode({
    id: `ip_adapter_${id}`,
    type: 'ip_adapter',
    weight,
    method,
    ip_adapter_model: model,
    clip_vision_model: clipVisionModel,
    begin_step_percent: beginEndStepPct[0],
    end_step_percent: beginEndStepPct[1],
    image: {
      image_name: imageObject.image.name,
    },
  });
  g.addEdge(ipAdapter, 'ip_adapter', ipAdapterCollect, 'item');
};

export const isValidIPAdapter = (ipa: CanvasIPAdapterState, base: BaseModelType): boolean => {
  // Must be have a model that matches the current base and must have a control image
  const hasModel = Boolean(ipa.model);
  const modelMatchesBase = ipa.model?.base === base;
  const hasImage = Boolean(ipa.imageObject);
  return hasModel && modelMatchesBase && hasImage;
};
