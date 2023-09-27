import { RootState } from 'app/store/store';
import {
  IPAdapterInvocation,
  MetadataAccumulatorInvocation,
} from 'services/api/types';
import {
  IPAdapterField,
  IPAdapterMetadataItem,
  NonNullableGraph,
} from '../../types/types';
import {
  CANVAS_COHERENCE_DENOISE_LATENTS,
  IP_ADAPTER,
  METADATA_ACCUMULATOR,
} from './constants';
import { omit } from 'lodash-es';

export const addIPAdapterToLinearGraph = (
  state: RootState,
  graph: NonNullableGraph,
  baseNodeId: string
): void => {
  const { isIPAdapterEnabled, ipAdapterInfo } = state.controlNet;

  console.log('ipAdapterInfo', ipAdapterInfo);

  const metadataAccumulator = graph.nodes[METADATA_ACCUMULATOR] as
    | MetadataAccumulatorInvocation
    | undefined;

  if (isIPAdapterEnabled && ipAdapterInfo.model) {
    const ipAdapterNode: IPAdapterInvocation = {
      id: IP_ADAPTER,
      type: 'ip_adapter',
      is_intermediate: true,
      weight: ipAdapterInfo.weight,
      ip_adapter_model: {
        base_model: ipAdapterInfo.model?.base_model,
        model_name: ipAdapterInfo.model?.model_name,
      },
      begin_step_percent: ipAdapterInfo.beginStepPct,
      end_step_percent: ipAdapterInfo.endStepPct,
    };

    console.log(ipAdapterNode, 'what the actual fuck 1');

    if (ipAdapterInfo.adapterImage) {
      ipAdapterNode.image = {
        image_name: ipAdapterInfo.adapterImage.image_name,
      };
    } else {
      return;
    }

    console.log(ipAdapterNode, 'what the actual fuck 2');

    graph.nodes[ipAdapterNode.id] = ipAdapterNode as IPAdapterInvocation;
    console.log(ipAdapterInfo, 'ip adapter info');
    if (metadataAccumulator?.ipAdapters) {
      const ipAdapterField = {
        image: {
          image_name: ipAdapterInfo.adapterImage.image_name,
        },
        ip_adapter_model: {
          base_model: ipAdapterInfo.model?.base_model,
          model_name: ipAdapterInfo.model?.model_name,
        },
        weight: ipAdapterInfo.weight,
        begin_step_percent: ipAdapterInfo.beginStepPct,
        end_step_percent: ipAdapterInfo.endStepPct,
        some_random_shit: 'test',
      };

      metadataAccumulator.ipAdapters.push(ipAdapterField);
      console.log('metadataAccumulator', metadataAccumulator);
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
