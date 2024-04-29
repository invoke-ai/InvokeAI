import type { RootState } from 'app/store/store';
import { selectValidControlNets } from 'features/controlAdapters/store/controlAdaptersSlice';
import type { ControlAdapterProcessorType, ControlNetConfig } from 'features/controlAdapters/store/types';
import type { ImageField } from 'features/nodes/types/common';
import { isControlAdapterLayer } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { differenceWith, intersectionWith } from 'lodash-es';
import type {
  CollectInvocation,
  ControlNetInvocation,
  CoreMetadataInvocation,
  NonNullableGraph,
  S,
} from 'services/api/types';
import { assert } from 'tsafe';

import { CONTROL_NET_COLLECT } from './constants';
import { upsertMetadata } from './metadata';

const getControlNets = (state: RootState) => {
  // Start with the valid controlnets
  const validControlNets = selectValidControlNets(state.controlAdapters).filter(
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

  // Collect all ControlNet ids for ControlNet layers
  const layerControlNetIds = state.regionalPrompts.present.layers
    .filter(isControlAdapterLayer)
    .map((l) => l.controlNetId);

  if (activeTabName === 'txt2img') {
    // Add only the cnets that are used in control layers
    return intersectionWith(validControlNets, layerControlNetIds, (a, b) => a.id === b);
  } else {
    // Else, we want to exclude the cnets that are used in control layers
    return differenceWith(validControlNets, layerControlNetIds, (a, b) => a.id === b);
  }
};

export const addControlNetToLinearGraph = async (
  state: RootState,
  graph: NonNullableGraph,
  baseNodeId: string
): Promise<void> => {
  const controlNets = getControlNets(state);
  const controlNetMetadata: CoreMetadataInvocation['controlnets'] = [];

  if (controlNets.length) {
    // Even though denoise_latents' control input is collection or scalar, keep it simple and always use a collect
    const controlNetIterateNode: CollectInvocation = {
      id: CONTROL_NET_COLLECT,
      type: 'collect',
      is_intermediate: true,
    };
    graph.nodes[CONTROL_NET_COLLECT] = controlNetIterateNode;
    graph.edges.push({
      source: { node_id: CONTROL_NET_COLLECT, field: 'collection' },
      destination: {
        node_id: baseNodeId,
        field: 'control',
      },
    });

    for (const controlNet of controlNets) {
      if (!controlNet.model) {
        return;
      }
      const {
        id,
        controlImage,
        processedControlImage,
        beginStepPct,
        endStepPct,
        controlMode,
        resizeMode,
        model,
        processorType,
        weight,
      } = controlNet;

      const controlNetNode: ControlNetInvocation = {
        id: `control_net_${id}`,
        type: 'controlnet',
        is_intermediate: true,
        begin_step_percent: beginStepPct,
        end_step_percent: endStepPct,
        control_mode: controlMode,
        resize_mode: resizeMode,
        control_model: model,
        control_weight: weight,
        image: buildControlImage(controlImage, processedControlImage, processorType),
      };

      graph.nodes[controlNetNode.id] = controlNetNode;

      controlNetMetadata.push(buildControlNetMetadata(controlNet));

      graph.edges.push({
        source: { node_id: controlNetNode.id, field: 'control' },
        destination: {
          node_id: CONTROL_NET_COLLECT,
          field: 'item',
        },
      });
    }
    upsertMetadata(graph, { controlnets: controlNetMetadata });
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
  assert(image, 'ControlNet image is required');
  return image;
};

const buildControlNetMetadata = (controlNet: ControlNetConfig): S['ControlNetMetadataField'] => {
  const {
    controlImage,
    processedControlImage,
    beginStepPct,
    endStepPct,
    controlMode,
    resizeMode,
    model,
    processorType,
    weight,
  } = controlNet;

  assert(model, 'ControlNet model is required');

  const processed_image =
    processedControlImage && processorType !== 'none'
      ? {
          image_name: processedControlImage,
        }
      : null;

  assert(controlImage, 'ControlNet image is required');

  return {
    control_model: model,
    control_weight: weight,
    control_mode: controlMode,
    begin_step_percent: beginStepPct,
    end_step_percent: endStepPct,
    resize_mode: resizeMode,
    image: {
      image_name: controlImage,
    },
    processed_image,
  };
};
