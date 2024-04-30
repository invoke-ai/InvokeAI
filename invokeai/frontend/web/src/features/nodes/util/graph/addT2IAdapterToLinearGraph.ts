import type { RootState } from 'app/store/store';
import { selectValidT2IAdapters } from 'features/controlAdapters/store/controlAdaptersSlice';
import type { ControlAdapterProcessorType, T2IAdapterConfig } from 'features/controlAdapters/store/types';
import { isControlAdapterLayer } from 'features/controlLayers/store/controlLayersSlice';
import type { ImageField } from 'features/nodes/types/common';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { differenceWith, intersectionWith } from 'lodash-es';
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

const getT2IAdapters = (state: RootState) => {
  // Start with the valid controlnets
  const validT2IAdapters = selectValidT2IAdapters(state.controlAdapters).filter(
    ({ model, processedControlImage, processorType, controlImage, isEnabled }) => {
      const hasModel = Boolean(model);
      const doesBaseMatch = model?.base === state.generation.model?.base;
      const hasControlImage = (processedControlImage && processorType !== 'none') || controlImage;

      return isEnabled && hasModel && doesBaseMatch && hasControlImage;
    }
  );

  // txt2img tab has special handling - it uses layers exclusively, while the other tabs use the older control adapters
  // accordion. We need to filter the list of valid T2I adapters according to the tab.
  const activeTabName = activeTabNameSelector(state);

  if (activeTabName === 'txt2img') {
    // Add only the T2Is that are used in control layers
    // Collect all ids for enabled control adapter layers
    const layerControlAdapterIds = state.controlLayers.present.layers
      .filter(isControlAdapterLayer)
      .filter((l) => l.isEnabled)
      .map((l) => l.controlNetId);
    return intersectionWith(validT2IAdapters, layerControlAdapterIds, (a, b) => a.id === b);
  } else {
    // Else, we want to exclude the T2Is that are used in control layers
    const layerControlAdapterIds = state.controlLayers.present.layers
      .filter(isControlAdapterLayer)
      .map((l) => l.controlNetId);
    return differenceWith(validT2IAdapters, layerControlAdapterIds, (a, b) => a.id === b);
  }
};

export const addT2IAdaptersToLinearGraph = async (
  state: RootState,
  graph: NonNullableGraph,
  baseNodeId: string
): Promise<void> => {
  const t2iAdapters = getT2IAdapters(state);

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
