import { RootState } from 'app/store/store';
import {
  IPAdapterInvocation,
  MetadataAccumulatorInvocation,
} from 'services/api/types';
import { NonNullableGraph } from '../../types/types';
import {
  CANVAS_COHERENCE_DENOISE_LATENTS,
  IP_ADAPTER,
  METADATA_ACCUMULATOR,
} from './constants';
import { selectValidIPAdapters } from 'features/controlNet/store/controlAdaptersSlice';

export const addIPAdapterToLinearGraph = (
  state: RootState,
  graph: NonNullableGraph,
  baseNodeId: string
): void => {
  const validIPAdapters = selectValidIPAdapters(state.controlAdapters);

  const metadataAccumulator = graph.nodes[METADATA_ACCUMULATOR] as
    | MetadataAccumulatorInvocation
    | undefined;

  const ipAdapter = validIPAdapters[0];

  // TODO: handle multiple IP adapters once backend is capable
  if (ipAdapter && ipAdapter.model) {
    const { weight, model, beginStepPct, endStepPct } = ipAdapter;
    const ipAdapterNode: IPAdapterInvocation = {
      id: IP_ADAPTER,
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
        node_id: baseNodeId,
        field: 'ip_adapter',
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
  }
};
