import { getCAId, getImageObjectId, getIPAId, getLayerId } from 'features/controlLayers/konva/naming';
import { defaultLoRAConfig } from 'features/controlLayers/store/lorasReducers';
import type { CanvasControlAdapterState, CanvasIPAdapterState, CanvasLayerState, LoRA } from 'features/controlLayers/store/types';
import {
  CA_PROCESSOR_DATA,
  imageDTOToImageWithDims,
  initialControlNetV2,
  initialIPAdapterV2,
  initialT2IAdapterV2,
  isProcessorTypeV2,
  zCanvasLayerState,
} from 'features/controlLayers/store/types';
import type {
  ControlNetConfigMetadata,
  IPAdapterConfigMetadata,
  MetadataParseFunc,
  T2IAdapterConfigMetadata,
} from 'features/metadata/types';
import { fetchModelConfigWithTypeGuard, getModelKey } from 'features/metadata/util/modelFetchingHelpers';
import { zControlField, zIPAdapterField, zModelIdentifierField, zT2IAdapterField } from 'features/nodes/types/common';
import type {
  ParameterCFGRescaleMultiplier,
  ParameterCFGScale,
  ParameterHeight,
  ParameterHRFEnabled,
  ParameterHRFMethod,
  ParameterModel,
  ParameterNegativePrompt,
  ParameterNegativeStylePromptSDXL,
  ParameterPositivePrompt,
  ParameterPositiveStylePromptSDXL,
  ParameterScheduler,
  ParameterSDXLRefinerModel,
  ParameterSDXLRefinerNegativeAestheticScore,
  ParameterSDXLRefinerPositiveAestheticScore,
  ParameterSDXLRefinerStart,
  ParameterSeed,
  ParameterSteps,
  ParameterStrength,
  ParameterVAEModel,
  ParameterWidth,
} from 'features/parameters/types/parameterSchemas';
import {
  isParameterCFGRescaleMultiplier,
  isParameterCFGScale,
  isParameterHeight,
  isParameterHRFEnabled,
  isParameterHRFMethod,
  isParameterLoRAWeight,
  isParameterNegativePrompt,
  isParameterNegativeStylePromptSDXL,
  isParameterPositivePrompt,
  isParameterPositiveStylePromptSDXL,
  isParameterScheduler,
  isParameterSDXLRefinerNegativeAestheticScore,
  isParameterSDXLRefinerPositiveAestheticScore,
  isParameterSDXLRefinerStart,
  isParameterSeed,
  isParameterSteps,
  isParameterStrength,
  isParameterWidth,
} from 'features/parameters/types/parameterSchemas';
import { get, isArray, isString } from 'lodash-es';
import { getImageDTO } from 'services/api/endpoints/images';
import {
  isControlNetModelConfig,
  isIPAdapterModelConfig,
  isLoRAModelConfig,
  isNonRefinerMainModelConfig,
  isRefinerMainModelModelConfig,
  isT2IAdapterModelConfig,
  isVAEModelConfig,
} from 'services/api/types';
import { assert } from 'tsafe';
import { v4 as uuidv4 } from 'uuid';

export const MetadataParsePendingToken = Symbol('pending');
export const MetadataParseFailedToken = Symbol('failed');
/**
 * Raised when metadata parsing fails.
 */
class MetadataParseError extends Error {
  /**
   * Create MetadataParseError
   * @param {String} message
   */
  constructor(message: string) {
    super(message);
    this.name = this.constructor.name;
  }
}

/**
 * An async function that a property from an object and validates its type using a type guard. If the property is missing
 * or invalid, the function should throw a MetadataParseError.
 * @param obj The object to get the property from.
 * @param property The property to get.
 * @param typeGuard A type guard function to check the type of the property. Provide `undefined` to opt out of type
 * validation and always return the property value.
 * @returns A promise that resolves to the property value if it exists and is of the expected type.
 * @throws MetadataParseError if a type guard is provided and the property is not of the expected type.
 */
const getProperty = <T = unknown>(
  obj: unknown,
  property: string,
  typeGuard: (val: unknown) => val is T = (val: unknown): val is T => true
): Promise<T> => {
  return new Promise<T>((resolve, reject) => {
    const val = get(obj, property) as unknown;
    if (typeGuard(val)) {
      resolve(val);
    }
    reject(new MetadataParseError(`Property ${property} is not of expected type`));
  });
};

const parseCreatedBy: MetadataParseFunc<string> = (metadata) => getProperty(metadata, 'created_by', isString);

const parseGenerationMode: MetadataParseFunc<string> = (metadata) => getProperty(metadata, 'generation_mode', isString);

const parsePositivePrompt: MetadataParseFunc<ParameterPositivePrompt> = (metadata) =>
  getProperty(metadata, 'positive_prompt', isParameterPositivePrompt);

const parseNegativePrompt: MetadataParseFunc<ParameterNegativePrompt> = (metadata) =>
  getProperty(metadata, 'negative_prompt', isParameterNegativePrompt);

const parseSDXLPositiveStylePrompt: MetadataParseFunc<ParameterPositiveStylePromptSDXL> = (metadata) =>
  getProperty(metadata, 'positive_style_prompt', isParameterPositiveStylePromptSDXL);

const parseSDXLNegativeStylePrompt: MetadataParseFunc<ParameterNegativeStylePromptSDXL> = (metadata) =>
  getProperty(metadata, 'negative_style_prompt', isParameterNegativeStylePromptSDXL);

const parseSeed: MetadataParseFunc<ParameterSeed> = (metadata) => getProperty(metadata, 'seed', isParameterSeed);

const parseCFGScale: MetadataParseFunc<ParameterCFGScale> = (metadata) =>
  getProperty(metadata, 'cfg_scale', isParameterCFGScale);

const parseCFGRescaleMultiplier: MetadataParseFunc<ParameterCFGRescaleMultiplier> = (metadata) =>
  getProperty(metadata, 'cfg_rescale_multiplier', isParameterCFGRescaleMultiplier);

const parseScheduler: MetadataParseFunc<ParameterScheduler> = (metadata) =>
  getProperty(metadata, 'scheduler', isParameterScheduler);

const parseWidth: MetadataParseFunc<ParameterWidth> = (metadata) => getProperty(metadata, 'width', isParameterWidth);

const parseHeight: MetadataParseFunc<ParameterHeight> = (metadata) =>
  getProperty(metadata, 'height', isParameterHeight);

const parseSteps: MetadataParseFunc<ParameterSteps> = (metadata) => getProperty(metadata, 'steps', isParameterSteps);

const parseStrength: MetadataParseFunc<ParameterStrength> = (metadata) =>
  getProperty(metadata, 'strength', isParameterStrength);

const parseHRFEnabled: MetadataParseFunc<ParameterHRFEnabled> = async (metadata) => {
  try {
    return await getProperty(metadata, 'hrf_enabled', isParameterHRFEnabled);
  } catch {
    return false;
  }
};

const parseHRFStrength: MetadataParseFunc<ParameterStrength> = (metadata) =>
  getProperty(metadata, 'hrf_strength', isParameterStrength);

const parseHRFMethod: MetadataParseFunc<ParameterHRFMethod> = (metadata) =>
  getProperty(metadata, 'hrf_method', isParameterHRFMethod);

const parseRefinerSteps: MetadataParseFunc<ParameterSteps> = (metadata) =>
  getProperty(metadata, 'refiner_steps', isParameterSteps);

const parseRefinerCFGScale: MetadataParseFunc<ParameterCFGScale> = (metadata) =>
  getProperty(metadata, 'refiner_cfg_scale', isParameterCFGScale);

const parseRefinerScheduler: MetadataParseFunc<ParameterScheduler> = (metadata) =>
  getProperty(metadata, 'refiner_scheduler', isParameterScheduler);

const parseRefinerPositiveAestheticScore: MetadataParseFunc<ParameterSDXLRefinerPositiveAestheticScore> = (metadata) =>
  getProperty(metadata, 'refiner_positive_aesthetic_score', isParameterSDXLRefinerPositiveAestheticScore);

const parseRefinerNegativeAestheticScore: MetadataParseFunc<ParameterSDXLRefinerNegativeAestheticScore> = (metadata) =>
  getProperty(metadata, 'refiner_negative_aesthetic_score', isParameterSDXLRefinerNegativeAestheticScore);

const parseRefinerStart: MetadataParseFunc<ParameterSDXLRefinerStart> = (metadata) =>
  getProperty(metadata, 'refiner_start', isParameterSDXLRefinerStart);

const parseMainModel: MetadataParseFunc<ParameterModel> = async (metadata) => {
  const model = await getProperty(metadata, 'model', undefined);
  const key = await getModelKey(model, 'main');
  const mainModelConfig = await fetchModelConfigWithTypeGuard(key, isNonRefinerMainModelConfig);
  const modelIdentifier = zModelIdentifierField.parse(mainModelConfig);
  return modelIdentifier;
};

const parseRefinerModel: MetadataParseFunc<ParameterSDXLRefinerModel> = async (metadata) => {
  const refiner_model = await getProperty(metadata, 'refiner_model', undefined);
  const key = await getModelKey(refiner_model, 'main');
  const refinerModelConfig = await fetchModelConfigWithTypeGuard(key, isRefinerMainModelModelConfig);
  const modelIdentifier = zModelIdentifierField.parse(refinerModelConfig);
  return modelIdentifier;
};

const parseVAEModel: MetadataParseFunc<ParameterVAEModel> = async (metadata) => {
  const vae = await getProperty(metadata, 'vae', undefined);
  const key = await getModelKey(vae, 'vae');
  const vaeModelConfig = await fetchModelConfigWithTypeGuard(key, isVAEModelConfig);
  const modelIdentifier = zModelIdentifierField.parse(vaeModelConfig);
  return modelIdentifier;
};

const parseLoRA: MetadataParseFunc<LoRA> = async (metadataItem) => {
  // Previously, the LoRA model identifier parts were stored in the LoRA metadata: `{key: ..., weight: 0.75}`
  const modelV1 = await getProperty(metadataItem, 'lora', undefined);
  // Now, the LoRA model is stored in a `model` property of the LoRA metadata: `{model: {key: ...}, weight: 0.75}`
  const modelV2 = await getProperty(metadataItem, 'model', undefined);
  const weight = await getProperty(metadataItem, 'weight', undefined);
  const key = await getModelKey(modelV2 ?? modelV1, 'lora');
  const loraModelConfig = await fetchModelConfigWithTypeGuard(key, isLoRAModelConfig);

  return {
    model: zModelIdentifierField.parse(loraModelConfig),
    weight: isParameterLoRAWeight(weight) ? weight : defaultLoRAConfig.weight,
    isEnabled: true,
  };
};

const parseAllLoRAs: MetadataParseFunc<LoRA[]> = async (metadata) => {
  try {
    const lorasRaw = await getProperty(metadata, 'loras', isArray);
    const parseResults = await Promise.allSettled(lorasRaw.map((lora) => parseLoRA(lora)));
    const loras = parseResults
      .filter((result): result is PromiseFulfilledResult<LoRA> => result.status === 'fulfilled')
      .map((result) => result.value);
    return loras;
  } catch {
    return [];
  }
};

const parseControlNet: MetadataParseFunc<ControlNetConfigMetadata> = async (metadataItem) => {
  const control_model = await getProperty(metadataItem, 'control_model');
  const key = await getModelKey(control_model, 'controlnet');
  const controlNetModel = await fetchModelConfigWithTypeGuard(key, isControlNetModelConfig);
  const image = zControlField.shape.image
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'image'));
  const processedImage = zControlField.shape.image
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'processed_image'));
  const control_weight = zControlField.shape.control_weight
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'control_weight'));
  const begin_step_percent = zControlField.shape.begin_step_percent
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'begin_step_percent'));
  const end_step_percent = zControlField.shape.end_step_percent
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'end_step_percent'));
  const control_mode = zControlField.shape.control_mode
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'control_mode'));
  const resize_mode = zControlField.shape.resize_mode
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'resize_mode'));

  const { processorType, processorNode } = buildControlAdapterProcessor(controlNetModel);

  const controlNet: ControlNetConfigMetadata = {
    type: 'controlnet',
    isEnabled: true,
    model: zModelIdentifierField.parse(controlNetModel),
    weight: typeof control_weight === 'number' ? control_weight : initialControlNet.weight,
    beginStepPct: begin_step_percent ?? initialControlNet.beginStepPct,
    endStepPct: end_step_percent ?? initialControlNet.endStepPct,
    controlMode: control_mode ?? initialControlNet.controlMode,
    resizeMode: resize_mode ?? initialControlNet.resizeMode,
    controlImage: image?.image_name ?? null,
    processedControlImage: processedImage?.image_name ?? null,
    processorType,
    processorNode,
    shouldAutoConfig: true,
    id: uuidv4(),
  };

  return controlNet;
};

const parseAllControlNets: MetadataParseFunc<ControlNetConfigMetadata[]> = async (metadata) => {
  try {
    const controlNetsRaw = await getProperty(metadata, 'controlnets', isArray);
    const parseResults = await Promise.allSettled(controlNetsRaw.map((cn) => parseControlNet(cn)));
    const controlNets = parseResults
      .filter((result): result is PromiseFulfilledResult<ControlNetConfigMetadata> => result.status === 'fulfilled')
      .map((result) => result.value);
    return controlNets;
  } catch {
    return [];
  }
};

const parseT2IAdapter: MetadataParseFunc<T2IAdapterConfigMetadata> = async (metadataItem) => {
  const t2i_adapter_model = await getProperty(metadataItem, 't2i_adapter_model');
  const key = await getModelKey(t2i_adapter_model, 't2i_adapter');
  const t2iAdapterModel = await fetchModelConfigWithTypeGuard(key, isT2IAdapterModelConfig);

  const image = zT2IAdapterField.shape.image
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'image'));
  const processedImage = zT2IAdapterField.shape.image
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'processed_image'));
  const weight = zT2IAdapterField.shape.weight
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'weight'));
  const begin_step_percent = zT2IAdapterField.shape.begin_step_percent
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'begin_step_percent'));
  const end_step_percent = zT2IAdapterField.shape.end_step_percent
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'end_step_percent'));
  const resize_mode = zT2IAdapterField.shape.resize_mode
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'resize_mode'));

  const { processorType, processorNode } = buildControlAdapterProcessor(t2iAdapterModel);

  const t2iAdapter: T2IAdapterConfigMetadata = {
    type: 't2i_adapter',
    isEnabled: true,
    model: zModelIdentifierField.parse(t2iAdapterModel),
    weight: typeof weight === 'number' ? weight : initialT2IAdapter.weight,
    beginStepPct: begin_step_percent ?? initialT2IAdapter.beginStepPct,
    endStepPct: end_step_percent ?? initialT2IAdapter.endStepPct,
    resizeMode: resize_mode ?? initialT2IAdapter.resizeMode,
    controlImage: image?.image_name ?? null,
    processedControlImage: processedImage?.image_name ?? null,
    processorType,
    processorNode,
    shouldAutoConfig: true,
    id: uuidv4(),
  };

  return t2iAdapter;
};

const parseAllT2IAdapters: MetadataParseFunc<T2IAdapterConfigMetadata[]> = async (metadata) => {
  try {
    const t2iAdaptersRaw = await getProperty(metadata, 't2iAdapters', isArray);
    const parseResults = await Promise.allSettled(t2iAdaptersRaw.map((t2iAdapter) => parseT2IAdapter(t2iAdapter)));
    const t2iAdapters = parseResults
      .filter((result): result is PromiseFulfilledResult<T2IAdapterConfigMetadata> => result.status === 'fulfilled')
      .map((result) => result.value);
    return t2iAdapters;
  } catch {
    return [];
  }
};

const parseIPAdapter: MetadataParseFunc<IPAdapterConfigMetadata> = async (metadataItem) => {
  const ip_adapter_model = await getProperty(metadataItem, 'ip_adapter_model');
  const key = await getModelKey(ip_adapter_model, 'ip_adapter');
  const ipAdapterModel = await fetchModelConfigWithTypeGuard(key, isIPAdapterModelConfig);

  const image = zIPAdapterField.shape.image
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'image'));
  const weight = zIPAdapterField.shape.weight
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'weight'));
  const method = zIPAdapterField.shape.method
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'method'));
  const begin_step_percent = zIPAdapterField.shape.begin_step_percent
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'begin_step_percent'));
  const end_step_percent = zIPAdapterField.shape.end_step_percent
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'end_step_percent'));

  const ipAdapter: IPAdapterConfigMetadata = {
    id: uuidv4(),
    type: 'ip_adapter',
    isEnabled: true,
    model: zModelIdentifierField.parse(ipAdapterModel),
    clipVisionModel: 'ViT-H',
    controlImage: image?.image_name ?? null,
    weight: weight ?? initialIPAdapter.weight,
    method: method ?? initialIPAdapter.method,
    beginStepPct: begin_step_percent ?? initialIPAdapter.beginStepPct,
    endStepPct: end_step_percent ?? initialIPAdapter.endStepPct,
  };

  return ipAdapter;
};

const parseAllIPAdapters: MetadataParseFunc<IPAdapterConfigMetadata[]> = async (metadata) => {
  try {
    const ipAdaptersRaw = await getProperty(metadata, 'ipAdapters', isArray);
    const parseResults = await Promise.allSettled(ipAdaptersRaw.map((ipAdapter) => parseIPAdapter(ipAdapter)));
    const ipAdapters = parseResults
      .filter((result): result is PromiseFulfilledResult<IPAdapterConfigMetadata> => result.status === 'fulfilled')
      .map((result) => result.value);
    return ipAdapters;
  } catch {
    return [];
  }
};

//#region Control Layers
const parseLayer: MetadataParseFunc<CanvasLayerState> = async (metadataItem) => zCanvasLayerState.parseAsync(metadataItem);

const parseLayers: MetadataParseFunc<CanvasLayerState[]> = async (metadata) => {
  // We need to support recalling pre-Control Layers metadata into Control Layers. A separate set of parsers handles
  // taking pre-CL metadata and parsing it into layers. It doesn't always map 1-to-1, so this is best-effort. For
  // example, CL Control Adapters don't support resize mode, so we simply omit that property.

  try {
    const layers: CanvasLayerState[] = [];

    try {
      const control_layers = await getProperty(metadata, 'control_layers');
      const controlLayersRaw = await getProperty(control_layers, 'layers', isArray);
      const controlLayersParseResults = await Promise.allSettled(controlLayersRaw.map(parseLayer));
      const controlLayers = controlLayersParseResults
        .filter((result): result is PromiseFulfilledResult<CanvasLayerState> => result.status === 'fulfilled')
        .map((result) => result.value);
      layers.push(...controlLayers);
    } catch {
      // no-op
    }

    try {
      const controlNetsRaw = await getProperty(metadata, 'controlnets', isArray);
      const controlNetsParseResults = await Promise.allSettled(
        controlNetsRaw.map(async (cn) => await parseControlNetToControlAdapterLayer(cn))
      );
      const controlNetsAsLayers = controlNetsParseResults
        .filter((result): result is PromiseFulfilledResult<CanvasControlAdapterState> => result.status === 'fulfilled')
        .map((result) => result.value);
      layers.push(...controlNetsAsLayers);
    } catch {
      // no-op
    }

    try {
      const t2iAdaptersRaw = await getProperty(metadata, 't2iAdapters', isArray);
      const t2iAdaptersParseResults = await Promise.allSettled(
        t2iAdaptersRaw.map(async (cn) => await parseT2IAdapterToControlAdapterLayer(cn))
      );
      const t2iAdaptersAsLayers = t2iAdaptersParseResults
        .filter((result): result is PromiseFulfilledResult<CanvasControlAdapterState> => result.status === 'fulfilled')
        .map((result) => result.value);
      layers.push(...t2iAdaptersAsLayers);
    } catch {
      // no-op
    }

    try {
      const ipAdaptersRaw = await getProperty(metadata, 'ipAdapters', isArray);
      const ipAdaptersParseResults = await Promise.allSettled(
        ipAdaptersRaw.map(async (cn) => await parseIPAdapterToIPAdapterLayer(cn))
      );
      const ipAdaptersAsLayers = ipAdaptersParseResults
        .filter((result): result is PromiseFulfilledResult<CanvasIPAdapterState> => result.status === 'fulfilled')
        .map((result) => result.value);
      layers.push(...ipAdaptersAsLayers);
    } catch {
      // no-op
    }

    try {
      const initialImageLayer = await parseInitialImageToInitialImageLayer(metadata);
      layers.push(initialImageLayer);
    } catch {
      // no-op
    }

    return layers;
  } catch {
    return [];
  }
};

const parseInitialImageToInitialImageLayer: MetadataParseFunc<CanvasLayerState> = async (metadata) => {
  // TODO(psyche): recall denoise strength
  // const denoisingStrength = await getProperty(metadata, 'strength', isParameterStrength);
  const imageName = await getProperty(metadata, 'init_image', isString);
  const imageDTO = await getImageDTO(imageName);
  assert(imageDTO, 'ImageDTO is null');
  const id = getLayerId(uuidv4());
  const layer: CanvasLayerState = {
    id,
    type: 'layer',
    bbox: null,
    bboxNeedsUpdate: true,
    x: 0,
    y: 0,
    isEnabled: true,
    opacity: 1,
    objects: [
      {
        type: 'image',
        id: getImageObjectId(id, imageDTO.image_name),
        width: imageDTO.width,
        height: imageDTO.height,
        image: imageDTOToImageWithDims(imageDTO),
        x: 0,
        y: 0,
      },
    ],
  };
  return layer;
};

const parseControlNetToControlAdapterLayer: MetadataParseFunc<CanvasControlAdapterState> = async (metadataItem) => {
  const control_model = await getProperty(metadataItem, 'control_model');
  const key = await getModelKey(control_model, 'controlnet');
  const controlNetModel = await fetchModelConfigWithTypeGuard(key, isControlNetModelConfig);
  const image = zControlField.shape.image
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'image'));
  const processedImage = zControlField.shape.image
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'processed_image'));
  const control_weight = zControlField.shape.control_weight
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'control_weight'));
  const begin_step_percent = zControlField.shape.begin_step_percent
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'begin_step_percent'));
  const end_step_percent = zControlField.shape.end_step_percent
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'end_step_percent'));
  const control_mode = zControlField.shape.control_mode
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'control_mode'));

  const defaultPreprocessor = controlNetModel.default_settings?.preprocessor;
  const processorConfig = isProcessorTypeV2(defaultPreprocessor)
    ? CA_PROCESSOR_DATA[defaultPreprocessor].buildDefaults()
    : null;
  const beginEndStepPct: [number, number] = [
    begin_step_percent ?? initialControlNetV2.beginEndStepPct[0],
    end_step_percent ?? initialControlNetV2.beginEndStepPct[1],
  ];
  const imageDTO = image ? await getImageDTO(image.image_name) : null;
  const processedImageDTO = processedImage ? await getImageDTO(processedImage.image_name) : null;

  const layer: CanvasControlAdapterState = {
    id: getCAId(uuidv4()),
    type: 'control_adapter',
    bbox: null,
    bboxNeedsUpdate: true,
    isEnabled: true,
    opacity: 1,
    filter: 'LightnessToAlphaFilter',
    x: 0,
    y: 0,
    adapterType: 'controlnet',
    model: zModelIdentifierField.parse(controlNetModel),
    weight: typeof control_weight === 'number' ? control_weight : initialControlNetV2.weight,
    beginEndStepPct,
    controlMode: control_mode ?? initialControlNetV2.controlMode,
    image: imageDTO ? imageDTOToImageWithDims(imageDTO) : null,
    processedImage: processedImageDTO ? imageDTOToImageWithDims(processedImageDTO) : null,
    processorConfig,
    processorPendingBatchId: null,
  };

  return layer;
};

const parseT2IAdapterToControlAdapterLayer: MetadataParseFunc<CanvasControlAdapterState> = async (metadataItem) => {
  const t2i_adapter_model = await getProperty(metadataItem, 't2i_adapter_model');
  const key = await getModelKey(t2i_adapter_model, 't2i_adapter');
  const t2iAdapterModel = await fetchModelConfigWithTypeGuard(key, isT2IAdapterModelConfig);

  const image = zT2IAdapterField.shape.image
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'image'));
  const processedImage = zT2IAdapterField.shape.image
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'processed_image'));
  const weight = zT2IAdapterField.shape.weight
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'weight'));
  const begin_step_percent = zT2IAdapterField.shape.begin_step_percent
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'begin_step_percent'));
  const end_step_percent = zT2IAdapterField.shape.end_step_percent
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'end_step_percent'));

  const defaultPreprocessor = t2iAdapterModel.default_settings?.preprocessor;
  const processorConfig = isProcessorTypeV2(defaultPreprocessor)
    ? CA_PROCESSOR_DATA[defaultPreprocessor].buildDefaults()
    : null;
  const beginEndStepPct: [number, number] = [
    begin_step_percent ?? initialT2IAdapterV2.beginEndStepPct[0],
    end_step_percent ?? initialT2IAdapterV2.beginEndStepPct[1],
  ];
  const imageDTO = image ? await getImageDTO(image.image_name) : null;
  const processedImageDTO = processedImage ? await getImageDTO(processedImage.image_name) : null;

  const layer: CanvasControlAdapterState = {
    id: getCAId(uuidv4()),
    bbox: null,
    bboxNeedsUpdate: true,
    isEnabled: true,
    filter: 'LightnessToAlphaFilter',
    opacity: 1,
    type: 'control_adapter',
    x: 0,
    y: 0,
    adapterType: 't2i_adapter',
    model: zModelIdentifierField.parse(t2iAdapterModel),
    weight: typeof weight === 'number' ? weight : initialT2IAdapterV2.weight,
    beginEndStepPct,
    image: imageDTO ? imageDTOToImageWithDims(imageDTO) : null,
    processedImage: processedImageDTO ? imageDTOToImageWithDims(processedImageDTO) : null,
    processorConfig,
    processorPendingBatchId: null,
  };

  return layer;
};

const parseIPAdapterToIPAdapterLayer: MetadataParseFunc<CanvasIPAdapterState> = async (metadataItem) => {
  const ip_adapter_model = await getProperty(metadataItem, 'ip_adapter_model');
  const key = await getModelKey(ip_adapter_model, 'ip_adapter');
  const ipAdapterModel = await fetchModelConfigWithTypeGuard(key, isIPAdapterModelConfig);

  const image = zIPAdapterField.shape.image
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'image'));
  const weight = zIPAdapterField.shape.weight
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'weight'));
  const method = zIPAdapterField.shape.method
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'method'));
  const begin_step_percent = zIPAdapterField.shape.begin_step_percent
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'begin_step_percent'));
  const end_step_percent = zIPAdapterField.shape.end_step_percent
    .nullish()
    .catch(null)
    .parse(await getProperty(metadataItem, 'end_step_percent'));

  const beginEndStepPct: [number, number] = [
    begin_step_percent ?? initialIPAdapterV2.beginEndStepPct[0],
    end_step_percent ?? initialIPAdapterV2.beginEndStepPct[1],
  ];
  const imageDTO = image ? await getImageDTO(image.image_name) : null;

  const layer: CanvasIPAdapterState = {
    id: getIPAId(uuidv4()),
    type: 'ip_adapter',
    isEnabled: true,
    model: zModelIdentifierField.parse(ipAdapterModel),
    weight: typeof weight === 'number' ? weight : initialIPAdapterV2.weight,
    beginEndStepPct,
    imageObject: imageDTO ? imageDTOToImageWithDims(imageDTO) : null,
    clipVisionModel: initialIPAdapterV2.clipVisionModel, // TODO: This needs to be added to the zIPAdapterField...
    method: method ?? initialIPAdapterV2.method,
  };

  return layer;
};
//#endregion

export const parsers = {
  createdBy: parseCreatedBy,
  generationMode: parseGenerationMode,
  positivePrompt: parsePositivePrompt,
  negativePrompt: parseNegativePrompt,
  sdxlPositiveStylePrompt: parseSDXLPositiveStylePrompt,
  sdxlNegativeStylePrompt: parseSDXLNegativeStylePrompt,
  seed: parseSeed,
  cfgScale: parseCFGScale,
  cfgRescaleMultiplier: parseCFGRescaleMultiplier,
  scheduler: parseScheduler,
  width: parseWidth,
  height: parseHeight,
  steps: parseSteps,
  strength: parseStrength,
  hrfEnabled: parseHRFEnabled,
  hrfStrength: parseHRFStrength,
  hrfMethod: parseHRFMethod,
  refinerSteps: parseRefinerSteps,
  refinerCFGScale: parseRefinerCFGScale,
  refinerScheduler: parseRefinerScheduler,
  refinerPositiveAestheticScore: parseRefinerPositiveAestheticScore,
  refinerNegativeAestheticScore: parseRefinerNegativeAestheticScore,
  refinerStart: parseRefinerStart,
  mainModel: parseMainModel,
  refinerModel: parseRefinerModel,
  vaeModel: parseVAEModel,
  lora: parseLoRA,
  loras: parseAllLoRAs,
  controlNet: parseControlNet,
  controlNets: parseAllControlNets,
  t2iAdapter: parseT2IAdapter,
  t2iAdapters: parseAllT2IAdapters,
  ipAdapter: parseIPAdapter,
  ipAdapters: parseAllIPAdapters,
  controlNetToControlLayer: parseControlNetToControlAdapterLayer,
  t2iAdapterToControlAdapterLayer: parseT2IAdapterToControlAdapterLayer,
  ipAdapterToIPAdapterLayer: parseIPAdapterToIPAdapterLayer,
  layer: parseLayer,
  layers: parseLayers,
} as const;
