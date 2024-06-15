import type { RootState } from 'app/store/store';
import { selectValidIPAdapters } from 'features/controlAdapters/store/controlAdaptersSlice';
import type { IPAdapterConfig } from 'features/controlAdapters/store/types';
import type { ImageField } from 'features/nodes/types/common';
import { upsertMetadata } from 'features/nodes/util/graph/canvas/metadata';
import { IP_ADAPTER_COLLECT } from 'features/nodes/util/graph/constants';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import type { Invocation, NonNullableGraph, S } from 'services/api/types';
import { assert } from 'tsafe';

export const addIPAdapterToLinearGraph = async (
  state: RootState,
  graph: NonNullableGraph,
  baseNodeId: string
): Promise<void> => {
  // The generation tab has special handling - its control adapters are set up in the Control Layers graph helper.
  const activeTabName = activeTabNameSelector(state);
  assert(activeTabName !== 'generation', 'Tried to use addT2IAdaptersToLinearGraph on generation tab');

  const ipAdapters = selectValidIPAdapters(state.controlAdapters).filter(({ model, controlImage, isEnabled }) => {
    const hasModel = Boolean(model);
    const doesBaseMatch = model?.base === state.canvasV2.params.model?.base;
    const hasControlImage = controlImage;
    return isEnabled && hasModel && doesBaseMatch && hasControlImage;
  });

  if (ipAdapters.length) {
    // Even though denoise_latents' ip adapter input is SINGLE_OR_COLLECTION, keep it simple and always use a collect
    const ipAdapterCollectNode: Invocation<'collect'> = {
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

    const ipAdapterMetdata: S['CoreMetadataInvocation']['ipAdapters'] = [];

    for (const ipAdapter of ipAdapters) {
      if (!ipAdapter.model) {
        return;
      }
      const { id, weight, model, clipVisionModel, method, beginStepPct, endStepPct, controlImage } = ipAdapter;

      assert(controlImage, 'IP Adapter image is required');

      const ipAdapterNode: Invocation<'ip_adapter'> = {
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
