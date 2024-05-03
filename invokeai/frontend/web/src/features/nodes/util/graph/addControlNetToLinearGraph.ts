import type { RootState } from 'app/store/store';
import { selectValidControlNets } from 'features/controlAdapters/store/controlAdaptersSlice';
import type { ControlAdapterProcessorType, ControlNetConfig } from 'features/controlAdapters/store/types';
import type { ImageField } from 'features/nodes/types/common';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
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

export const addControlNetToLinearGraph = async (
  state: RootState,
  graph: NonNullableGraph,
  baseNodeId: string
): Promise<void> => {
  const controlNetMetadata: CoreMetadataInvocation['controlnets'] = [];
  const controlNets = selectValidControlNets(state.controlAdapters).filter(
    ({ model, processedControlImage, processorType, controlImage, isEnabled }) => {
      const hasModel = Boolean(model);
      const doesBaseMatch = model?.base === state.generation.model?.base;
      const hasControlImage = (processedControlImage && processorType !== 'none') || controlImage;

      return isEnabled && hasModel && doesBaseMatch && hasControlImage;
    }
  );

  // The txt2img tab has special handling - its control adapters are set up in the Control Layers graph helper.
  const activeTabName = activeTabNameSelector(state);
  assert(activeTabName !== 'txt2img', 'Tried to use addControlNetToLinearGraph on txt2img tab');

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
