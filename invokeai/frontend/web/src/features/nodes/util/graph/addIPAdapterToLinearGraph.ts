import type { RootState } from 'app/store/store';
import { selectValidIPAdapters } from 'features/controlAdapters/store/controlAdaptersSlice';
import { fetchModelConfigWithTypeGuard } from 'features/metadata/util/modelFetchingHelpers';
import type {
  CollectInvocation,
  CoreMetadataInvocation,
  IPAdapterInvocation,
  NonNullableGraph,
} from 'services/api/types';
import { isIPAdapterModelConfig } from 'services/api/types';

import { IP_ADAPTER_COLLECT } from './constants';
import { getModelMetadataField, upsertMetadata } from './metadata';

export const addIPAdapterToLinearGraph = async (
  state: RootState,
  graph: NonNullableGraph,
  baseNodeId: string
): Promise<void> => {
  const validIPAdapters = selectValidIPAdapters(state.controlAdapters).filter(({ model, controlImage, isEnabled }) => {
    const hasModel = Boolean(model);
    const doesBaseMatch = model?.base === state.generation.model?.base;
    const hasControlImage = controlImage;
    return isEnabled && hasModel && doesBaseMatch && hasControlImage;
  });

  if (validIPAdapters.length) {
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

    validIPAdapters.forEach(async (ipAdapter) => {
      if (!ipAdapter.model) {
        return;
      }
      const { id, weight, model, beginStepPct, endStepPct } = ipAdapter;
      const ipAdapterNode: IPAdapterInvocation = {
        id: `ip_adapter_${id}`,
        type: 'ip_adapter',
        is_intermediate: true,
        weight: weight,
        ip_adapter_model: model,
        begin_step_percent: beginStepPct,
        end_step_percent: endStepPct,
      };

      if (ipAdapter.controlImage) {
        ipAdapterNode.image = {
          image_name: ipAdapter.controlImage,
        };
      } else {
        return;
      }

      graph.nodes[ipAdapterNode.id] = ipAdapterNode;

      const modelConfig = await fetchModelConfigWithTypeGuard(model.key, isIPAdapterModelConfig);

      ipAdapterMetdata.push({
        weight: weight,
        ip_adapter_model: getModelMetadataField(modelConfig),
        begin_step_percent: beginStepPct,
        end_step_percent: endStepPct,
        image: ipAdapterNode.image,
      });

      graph.edges.push({
        source: { node_id: ipAdapterNode.id, field: 'ip_adapter' },
        destination: {
          node_id: ipAdapterCollectNode.id,
          field: 'item',
        },
      });
    });

    upsertMetadata(graph, { ipAdapters: ipAdapterMetdata });
  }
};
