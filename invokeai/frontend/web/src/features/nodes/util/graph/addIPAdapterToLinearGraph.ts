import type { RootState } from 'app/store/store';
import { selectValidIPAdapters } from 'features/controlAdapters/store/controlAdaptersSlice';
import type { IPAdapterConfig } from 'features/controlAdapters/store/types';
import type { ImageField } from 'features/nodes/types/common';
import { isIPAdapterLayer, isMaskedGuidanceLayer } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { differenceWith, intersectionWith } from 'lodash-es';
import type {
  CollectInvocation,
  CoreMetadataInvocation,
  IPAdapterInvocation,
  NonNullableGraph,
  S,
} from 'services/api/types';
import { assert } from 'tsafe';

import { IP_ADAPTER_COLLECT } from './constants';
import { upsertMetadata } from './metadata';

const getIPAdapters = (state: RootState) => {
  // Start with the valid IP adapters
  const validIPAdapters = selectValidIPAdapters(state.controlAdapters).filter(({ model, controlImage, isEnabled }) => {
    const hasModel = Boolean(model);
    const doesBaseMatch = model?.base === state.generation.model?.base;
    const hasControlImage = controlImage;
    return isEnabled && hasModel && doesBaseMatch && hasControlImage;
  });

  // Masked IP adapters are handled in the graph helper for regional control - skip them here
  const maskedIPAdapterIds = state.regionalPrompts.present.layers
    .filter(isMaskedGuidanceLayer)
    .map((l) => l.ipAdapterIds)
    .flat();
  const nonMaskedIPAdapters = differenceWith(validIPAdapters, maskedIPAdapterIds, (a, b) => a.id === b);

  // txt2img tab has special handling - it uses layers exclusively, while the other tabs use the older control adapters
  // accordion. We need to filter the list of valid IP adapters according to the tab.
  const activeTabName = activeTabNameSelector(state);

  // Collect all IP Adapter ids for IP adapter layers
  const layerIPAdapterIds = state.regionalPrompts.present.layers.filter(isIPAdapterLayer).map((l) => l.ipAdapterId);

  if (activeTabName === 'txt2img') {
    // If we are on the t2i tab, we only want to add the IP adapters that are used in unmasked IP Adapter layers
    return intersectionWith(nonMaskedIPAdapters, layerIPAdapterIds, (a, b) => a.id === b);
  } else {
    // Else, we want to exclude the IP adapters that are used in IP Adapter layers
    return differenceWith(nonMaskedIPAdapters, layerIPAdapterIds, (a, b) => a.id === b);
  }
};

export const addIPAdapterToLinearGraph = async (
  state: RootState,
  graph: NonNullableGraph,
  baseNodeId: string
): Promise<void> => {
  const ipAdapters = getIPAdapters(state);

  if (ipAdapters.length) {
    // Even though denoise_latents' ip adapter input is collection or scalar, keep it simple and always use a collect
    const ipAdapterCollectNode: CollectInvocation = {
      id: IP_ADAPTER_COLLECT,
      type: 'collect',
      is_intermediate: true,
    };
    graph.nodes[IP_ADAPTER_COLLECT] = ipAdapterCollectNode;
    graph.edges.push({
      source: { node_id: IP_ADAPTER_COLLECT, field: 'collection' },
      destination: {
        node_id: baseNodeId,
        field: 'ip_adapter',
      },
    });

    const ipAdapterMetdata: CoreMetadataInvocation['ipAdapters'] = [];

    for (const ipAdapter of ipAdapters) {
      if (!ipAdapter.model) {
        return;
      }
      const { id, weight, model, clipVisionModel, method, beginStepPct, endStepPct, controlImage } = ipAdapter;

      assert(controlImage, 'IP Adapter image is required');

      const ipAdapterNode: IPAdapterInvocation = {
        id: `ip_adapter_${id}`,
        type: 'ip_adapter',
        is_intermediate: true,
        weight: weight,
        method: method,
        ip_adapter_model: model,
        clip_vision_model: clipVisionModel,
        begin_step_percent: beginStepPct,
        end_step_percent: endStepPct,
        image: {
          image_name: controlImage,
        },
      };

      graph.nodes[ipAdapterNode.id] = ipAdapterNode;

      ipAdapterMetdata.push(buildIPAdapterMetadata(ipAdapter));

      graph.edges.push({
        source: { node_id: ipAdapterNode.id, field: 'ip_adapter' },
        destination: {
          node_id: ipAdapterCollectNode.id,
          field: 'item',
        },
      });
    }

    upsertMetadata(graph, { ipAdapters: ipAdapterMetdata });
  }
};

const buildIPAdapterMetadata = (ipAdapter: IPAdapterConfig): S['IPAdapterMetadataField'] => {
  const { controlImage, beginStepPct, endStepPct, model, clipVisionModel, method, weight } = ipAdapter;

  assert(model, 'IP Adapter model is required');

  let image: ImageField | null = null;

  if (controlImage) {
    image = {
      image_name: controlImage,
    };
  }

  assert(image, 'IP Adapter image is required');

  return {
    ip_adapter_model: model,
    clip_vision_model: clipVisionModel,
    weight,
    method,
    begin_step_percent: beginStepPct,
    end_step_percent: endStepPct,
    image,
  };
};
