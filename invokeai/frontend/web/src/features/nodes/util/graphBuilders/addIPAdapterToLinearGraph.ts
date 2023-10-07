import { RootState } from 'app/store/store';
import { selectValidIPAdapters } from 'features/controlAdapters/store/controlAdaptersSlice';
import {
  CollectInvocation,
  IPAdapterInvocation,
  MetadataAccumulatorInvocation,
} from 'services/api/types';
import { NonNullableGraph } from '../../types/types';
import {
  CANVAS_COHERENCE_DENOISE_LATENTS,
  IP_ADAPTER_COLLECT,
  METADATA_ACCUMULATOR,
} from './constants';

export const addIPAdapterToLinearGraph = (
  state: RootState,
  graph: NonNullableGraph,
  baseNodeId: string
): void => {
  const validIPAdapters = selectValidIPAdapters(state.controlAdapters).filter(
    (ca) => ca.model?.base_model === state.generation.model?.base_model
  );

  const metadataAccumulator = graph.nodes[METADATA_ACCUMULATOR] as
    | MetadataAccumulatorInvocation
    | undefined;

  if (validIPAdapters.length) {
    // Even though denoise_latents' control input is polymorphic, keep it simple and always use a collect
    const ipAdapterCollectNode: CollectInvocation = {
      id: IP_ADAPTER_COLLECT,
      type: 'collect',
      is_intermediate: true,
    };
    graph.nodes[ipAdapterCollectNode.id] = ipAdapterCollectNode;
    graph.edges.push({
      source: { node_id: ipAdapterCollectNode.id, field: 'collection' },
      destination: {
        node_id: baseNodeId,
        field: 'ip_adapter',
      },
    });

    validIPAdapters.forEach((ipAdapter) => {
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

      graph.nodes[ipAdapterNode.id] = ipAdapterNode as IPAdapterInvocation;

      if (metadataAccumulator?.ipAdapters) {
        const ipAdapterField = {
          image: {
            image_name: ipAdapter.controlImage,
          },
          weight,
          ip_adapter_model: model,
          begin_step_percent: beginStepPct,
          end_step_percent: endStepPct,
        };

        metadataAccumulator.ipAdapters.push(ipAdapterField);
      }

      graph.edges.push({
        source: { node_id: ipAdapterNode.id, field: 'ip_adapter' },
        destination: {
          node_id: ipAdapterCollectNode.id,
          field: 'item',
        },
      });

      if (CANVAS_COHERENCE_DENOISE_LATENTS in graph.nodes) {
        graph.edges.push({
          source: { node_id: ipAdapterNode.id, field: 'ip_adapter' },
          destination: {
            node_id: CANVAS_COHERENCE_DENOISE_LATENTS,
            field: 'ip_adapter',
          },
        });
      }
    });
  }
};
