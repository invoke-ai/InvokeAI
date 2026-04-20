import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectCanvasSettingsSlice } from 'features/controlLayers/store/canvasSettingsSlice';
import { selectModelConfig, selectParamsSlice } from 'features/controlLayers/store/paramsSlice';
import { selectRefImagesSlice } from 'features/controlLayers/store/refImagesSlice';
import { type ModelIdentifierField, zImageField, zModelIdentifierField } from 'features/nodes/types/common';
import { Graph } from 'features/nodes/util/graph/generation/Graph';
import {
  getOriginalAndScaledSizesForOtherModes,
  selectCanvasOutputFields,
} from 'features/nodes/util/graph/graphBuilderUtils';
import {
  type GraphBuilderArg,
  type GraphBuilderReturn,
  UnsupportedGenerationModeError,
} from 'features/nodes/util/graph/types';
import { hasExternalPanelControl } from 'features/parameters/util/externalPanelSchema';
import {
  type AnyInvocation,
  type AnyModelConfigWithExternal,
  type Invocation,
  isExternalApiModelConfig,
} from 'services/api/types';
import { assert } from 'tsafe';

const EXTERNAL_PROVIDER_NODE_TYPES = {
  gemini: 'gemini_image_generation',
  openai: 'openai_image_generation',
} as const;

export const buildExternalGraph = async (arg: GraphBuilderArg): Promise<GraphBuilderReturn> => {
  const { generationMode, state, manager } = arg;

  const modelConfig = selectModelConfig(state) as AnyModelConfigWithExternal | null;
  assert(modelConfig, 'No model selected');
  assert(isExternalApiModelConfig(modelConfig), 'Selected model is not an external API model');
  const model = modelConfig;

  if (generationMode === 'outpaint') {
    throw new UnsupportedGenerationModeError('Outpainting is not supported for external API models.');
  }
  const requestedMode = generationMode;
  if (!model.capabilities.modes.includes(requestedMode)) {
    throw new UnsupportedGenerationModeError(`${model.name} does not support ${requestedMode} mode`);
  }

  const params = selectParamsSlice(state);
  const refImages = selectRefImagesSlice(state);

  const g = new Graph(getPrefixedId('external_graph'));
  const supportsSeed = hasExternalPanelControl(model, 'image', 'seed');
  const supportsReferenceImages = hasExternalPanelControl(model, 'prompts', 'reference_images');

  const seed = supportsSeed
    ? g.addNode({
        id: getPrefixedId('seed'),
        type: 'integer',
      })
    : null;

  const positivePrompt = g.addNode({
    id: getPrefixedId('positive_prompt'),
    type: 'string',
  });

  const externalNodeType = EXTERNAL_PROVIDER_NODE_TYPES[model.provider_id as keyof typeof EXTERNAL_PROVIDER_NODE_TYPES];
  assert(externalNodeType, `No invocation node registered for external provider '${model.provider_id}'`);
  const externalNode: Record<string, unknown> = {
    id: getPrefixedId(externalNodeType),
    type: externalNodeType,
    model: model as unknown as ModelIdentifierField,
    mode: requestedMode,
    image_size: params.imageSize ?? null,
    num_images: 1,
  };

  // Provider-specific options
  if (model.provider_id === 'openai') {
    externalNode.quality = params.openaiQuality;
    externalNode.background = params.openaiBackground;
    if (params.openaiInputFidelity) {
      externalNode.input_fidelity = params.openaiInputFidelity;
    }
  } else if (model.provider_id === 'gemini') {
    if (params.geminiTemperature !== null) {
      externalNode.temperature = params.geminiTemperature;
    }
  }
  const externalInvocation = g.addNode(externalNode as AnyInvocation);

  if (seed) {
    g.addEdgeFromObj({
      source: { node_id: seed.id, field: 'value' },
      destination: { node_id: externalNode.id as string, field: 'seed' },
    });
  }
  g.addEdgeFromObj({
    source: { node_id: positivePrompt.id, field: 'value' },
    destination: { node_id: externalNode.id as string, field: 'prompt' },
  });

  if (supportsReferenceImages) {
    const referenceImages = refImages.entities
      .filter((entity) => entity.isEnabled)
      .map((entity) => entity.config)
      .filter((config) => config.image)
      .map((config) => zImageField.parse(config.image?.crop?.image ?? config.image?.original.image));

    if (referenceImages.length > 0) {
      externalNode.reference_images = referenceImages;
    }
  }

  // External models require specific dimensions matching their supported presets.
  // Always use params dimensions (from selected preset) for the API width/height.
  externalNode.width = params.dimensions.width;
  externalNode.height = params.dimensions.height;

  if (generationMode !== 'txt2img') {
    assert(manager, 'Canvas manager is required for img2img/inpaint');
    const canvasSettings = selectCanvasSettingsSlice(state);
    const { rect } = getOriginalAndScaledSizesForOtherModes(state);

    const rasterAdapters = manager.compositor.getVisibleAdaptersOfType('raster_layer');
    const initImage = await manager.compositor.getCompositeImageDTO(rasterAdapters, rect, {
      is_intermediate: true,
      silent: true,
    });
    externalNode.init_image = { image_name: initImage.image_name };

    if (generationMode === 'inpaint') {
      const inpaintMaskAdapters = manager.compositor.getVisibleAdaptersOfType('inpaint_mask');
      const maskImage = await manager.compositor.getGrayscaleMaskCompositeImageDTO(
        inpaintMaskAdapters,
        rect,
        'denoiseLimit',
        canvasSettings.preserveMask,
        {
          is_intermediate: true,
          silent: true,
        }
      );
      externalNode.mask_image = { image_name: maskImage.image_name };
    }
  }

  g.updateNode(externalNode as AnyInvocation, selectCanvasOutputFields(state));

  g.upsertMetadata({
    model: zModelIdentifierField.parse(model),
    width: externalNode.width as number,
    height: externalNode.height as number,
  });
  g.addEdgeToMetadata(positivePrompt as Invocation<'string'>, 'value', 'positive_prompt');
  if (seed) {
    g.addEdgeToMetadata(seed, 'value', 'seed');
  }

  g.setMetadataReceivingNode(externalInvocation);

  return {
    g,
    seed: seed ?? undefined,
    positivePrompt: positivePrompt as Invocation<'string'>,
  };
};
