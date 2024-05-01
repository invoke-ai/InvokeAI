import type { RootState } from 'app/store/store';
import { selectValidT2IAdapters } from 'features/controlAdapters/store/controlAdaptersSlice';
import type { ControlAdapterProcessorType, T2IAdapterConfig } from 'features/controlAdapters/store/types';
import type { ImageField } from 'features/nodes/types/common';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import type {
  CollectInvocation,
  CoreMetadataInvocation,
  NonNullableGraph,
  S,
  T2IAdapterInvocation,
} from 'services/api/types';
import { assert } from 'tsafe';

import { T2I_ADAPTER_COLLECT } from './constants';
import { upsertMetadata } from './metadata';

export const addT2IAdaptersToLinearGraph = async (
  state: RootState,
  graph: NonNullableGraph,
  baseNodeId: string
): Promise<void> => {
  // The txt2img tab has special handling - its control adapters are set up in the Control Layers graph helper.
  const activeTabName = activeTabNameSelector(state);
  assert(activeTabName !== 'txt2img', 'Tried to use addT2IAdaptersToLinearGraph on txt2img tab');

  const t2iAdapters = selectValidT2IAdapters(state.controlAdapters).filter(
    ({ model, processedControlImage, processorType, controlImage, isEnabled }) => {
      const hasModel = Boolean(model);
      const doesBaseMatch = model?.base === state.generation.model?.base;
      const hasControlImage = (processedControlImage && processorType !== 'none') || controlImage;

      return isEnabled && hasModel && doesBaseMatch && hasControlImage;
    }
  );

  if (t2iAdapters.length) {
    // Even though denoise_latents' t2i adapter input is collection or scalar, keep it simple and always use a collect
    const t2iAdapterCollectNode: CollectInvocation = {
      id: T2I_ADAPTER_COLLECT,
      type: 'collect',
      is_intermediate: true,
    };
    graph.nodes[T2I_ADAPTER_COLLECT] = t2iAdapterCollectNode;
    graph.edges.push({
      source: { node_id: T2I_ADAPTER_COLLECT, field: 'collection' },
      destination: {
        node_id: baseNodeId,
        field: 't2i_adapter',
      },
    });

    const t2iAdapterMetadata: CoreMetadataInvocation['t2iAdapters'] = [];

    for (const t2iAdapter of t2iAdapters) {
      if (!t2iAdapter.model) {
        return;
      }
      const {
        id,
        controlImage,
        processedControlImage,
        beginStepPct,
        endStepPct,
        resizeMode,
        model,
        processorType,
        weight,
      } = t2iAdapter;

      const t2iAdapterNode: T2IAdapterInvocation = {
        id: `t2i_adapter_${id}`,
        type: 't2i_adapter',
        is_intermediate: true,
        begin_step_percent: beginStepPct,
        end_step_percent: endStepPct,
        resize_mode: resizeMode,
        t2i_adapter_model: model,
        weight: weight,
        image: buildControlImage(controlImage, processedControlImage, processorType),
      };

      graph.nodes[t2iAdapterNode.id] = t2iAdapterNode;

      t2iAdapterMetadata.push(buildT2IAdapterMetadata(t2iAdapter));

      graph.edges.push({
        source: { node_id: t2iAdapterNode.id, field: 't2i_adapter' },
        destination: {
          node_id: T2I_ADAPTER_COLLECT,
          field: 'item',
        },
      });
    }

    upsertMetadata(graph, { t2iAdapters: t2iAdapterMetadata });
  }
};

const buildControlImage = (
  controlImage: string | null,
  processedControlImage: string | null,
  processorType: ControlAdapterProcessorType
): ImageField => {
  let image: ImageField | null = null;
  if (processedControlImage && processorType !== 'none') {
    // We've already processed the image in the app, so we can just use the processed image
    image = {
      image_name: processedControlImage,
    };
  } else if (controlImage) {
    // The control image is preprocessed
    image = {
      image_name: controlImage,
    };
  }
  assert(image, 'T2I Adapter image is required');
  return image;
};

const buildT2IAdapterMetadata = (t2iAdapter: T2IAdapterConfig): S['T2IAdapterMetadataField'] => {
  const { controlImage, processedControlImage, beginStepPct, endStepPct, resizeMode, model, processorType, weight } =
    t2iAdapter;

  assert(model, 'T2I Adapter model is required');

  const processed_image =
    processedControlImage && processorType !== 'none'
      ? {
          image_name: processedControlImage,
        }
      : null;

  assert(controlImage, 'T2I Adapter image is required');

  return {
    t2i_adapter_model: model,
    weight,
    begin_step_percent: beginStepPct,
    end_step_percent: endStepPct,
    resize_mode: resizeMode,
    image: {
      image_name: controlImage,
    },
    processed_image,
  };
};
