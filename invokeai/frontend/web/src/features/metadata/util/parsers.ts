import { getPrefixedId } from 'features/controlLayers/konva/util';
import { defaultLoRAConfig } from 'features/controlLayers/store/lorasSlice';
import type {
  CanvasControlLayerState,
  CanvasInpaintMaskState,
  CanvasMetadata,
  CanvasRasterLayerState,
  CanvasReferenceImageState,
  CanvasRegionalGuidanceState,
  LoRA,
} from 'features/controlLayers/store/types';
import { zCanvasMetadata, zCanvasRasterLayerState } from 'features/controlLayers/store/types';
import {
  imageDTOToImageWithDims,
  initialControlNet,
  initialIPAdapter,
  initialT2IAdapter,
} from 'features/controlLayers/store/util';
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
  ParameterGuidance,
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
  isParameterGuidance,
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
import { getImageDTOSafe } from 'services/api/endpoints/images';
import {
  isControlNetModelConfig,
  isIPAdapterModelConfig,
  isLoRAModelConfig,
  isNonRefinerMainModelConfig,
  isRefinerMainModelModelConfig,
  isT2IAdapterModelConfig,
  isVAEModelConfig,
} from 'services/api/types';

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

const parseGuidance: MetadataParseFunc<ParameterGuidance> = (metadata) =>
  getProperty(metadata, 'guidance', isParameterGuidance);

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
    id: getPrefixedId('lora'),
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

  const controlNet: ControlNetConfigMetadata = {
    type: 'controlnet',
    model: zModelIdentifierField.parse(controlNetModel),
    weight: typeof control_weight === 'number' ? control_weight : initialControlNet.weight,
    beginEndStepPct: [
      begin_step_percent ?? initialControlNet.beginEndStepPct[0],
      end_step_percent ?? initialControlNet.beginEndStepPct[1],
    ],
    controlMode: control_mode ?? initialControlNet.controlMode,
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

  const t2iAdapter: T2IAdapterConfigMetadata = {
    type: 't2i_adapter',
    model: zModelIdentifierField.parse(t2iAdapterModel),
    weight: typeof weight === 'number' ? weight : initialT2IAdapter.weight,
    beginEndStepPct: [
      begin_step_percent ?? initialT2IAdapter.beginEndStepPct[0],
      end_step_percent ?? initialT2IAdapter.beginEndStepPct[1],
    ],
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

  // const image = zIPAdapterField.shape.image
  //   .nullish()
  //   .catch(null)
  //   .parse(await getProperty(metadataItem, 'image'));
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
    type: 'ip_adapter',
    model: zModelIdentifierField.parse(ipAdapterModel),
    clipVisionModel: 'ViT-H',
    image: null, //TODO(psyche): need an ImageWithDims
    weight: weight ?? initialIPAdapter.weight,
    method: method ?? initialIPAdapter.method,
    beginEndStepPct: [
      begin_step_percent ?? initialIPAdapter.beginEndStepPct[0],
      end_step_percent ?? initialIPAdapter.beginEndStepPct[1],
    ],
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
const parseLayer: MetadataParseFunc<
  | CanvasRasterLayerState
  | CanvasControlLayerState
  | CanvasReferenceImageState
  | CanvasRegionalGuidanceState
  | CanvasInpaintMaskState
> = (metadataItem) => zCanvasRasterLayerState.parseAsync(metadataItem);

const parseCanvasV2Metadata: MetadataParseFunc<CanvasMetadata> = async (metadata) => {
  const metadataItem = await getProperty(metadata, 'canvas_v2_metadata', undefined);
  return zCanvasMetadata.parseAsync(metadataItem);
};

const parseLayers: MetadataParseFunc<
  (
    | CanvasRasterLayerState
    | CanvasControlLayerState
    | CanvasReferenceImageState
    | CanvasRegionalGuidanceState
    | CanvasInpaintMaskState
  )[]
> = async (metadata) => {
  // We need to support recalling pre-Control Layers metadata into Control Layers. A separate set of parsers handles
  // taking pre-CL metadata and parsing it into layers. It doesn't always map 1-to-1, so this is best-effort. For
  // example, CL Control Adapters don't support resize mode, so we simply omit that property.

  try {
    const layers: (
      | CanvasRasterLayerState
      | CanvasControlLayerState
      | CanvasReferenceImageState
      | CanvasRegionalGuidanceState
      | CanvasInpaintMaskState
    )[] = [];

    try {
      const control_layers = await getProperty(metadata, 'control_layers');
      const controlLayersRaw = await getProperty(control_layers, 'layers', isArray);
      const controlLayersParseResults = await Promise.allSettled(controlLayersRaw.map(parseLayer));
      const controlLayers = controlLayersParseResults
        .filter((result): result is PromiseFulfilledResult<CanvasRasterLayerState> => result.status === 'fulfilled')
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
        .filter((result): result is PromiseFulfilledResult<CanvasControlLayerState> => result.status === 'fulfilled')
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
        .filter((result): result is PromiseFulfilledResult<CanvasControlLayerState> => result.status === 'fulfilled')
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
        .filter((result): result is PromiseFulfilledResult<CanvasReferenceImageState> => result.status === 'fulfilled')
        .map((result) => result.value);
      layers.push(...ipAdaptersAsLayers);
    } catch {
      // no-op
    }

    return layers;
  } catch {
    return [];
  }
};

const parseControlNetToControlAdapterLayer: MetadataParseFunc<CanvasControlLayerState> = async (metadataItem) => {
  const control_model = await getProperty(metadataItem, 'control_model');
  const key = await getModelKey(control_model, 'controlnet');
  const controlNetModel = await fetchModelConfigWithTypeGuard(key, isControlNetModelConfig);
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

  const beginEndStepPct: [number, number] = [
    begin_step_percent ?? initialControlNet.beginEndStepPct[0],
    end_step_percent ?? initialControlNet.beginEndStepPct[1],
  ];

  const layer: CanvasControlLayerState = {
    id: getPrefixedId('control_layer'),
    type: 'control_layer',
    isEnabled: true,
    isLocked: false,
    opacity: 1,
    name: null,
    objects: [],
    position: { x: 0, y: 0 },
    withTransparencyEffect: false,
    controlAdapter: {
      type: 'controlnet',
      model: zModelIdentifierField.parse(controlNetModel),
      weight: typeof control_weight === 'number' ? control_weight : initialControlNet.weight,
      beginEndStepPct,
      controlMode: control_mode ?? initialControlNet.controlMode,
    },
  };

  return layer;
};

const parseT2IAdapterToControlAdapterLayer: MetadataParseFunc<CanvasControlLayerState> = async (metadataItem) => {
  const t2i_adapter_model = await getProperty(metadataItem, 't2i_adapter_model');
  const key = await getModelKey(t2i_adapter_model, 't2i_adapter');
  const t2iAdapterModel = await fetchModelConfigWithTypeGuard(key, isT2IAdapterModelConfig);

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

  const beginEndStepPct: [number, number] = [
    begin_step_percent ?? initialT2IAdapter.beginEndStepPct[0],
    end_step_percent ?? initialT2IAdapter.beginEndStepPct[1],
  ];

  const layer: CanvasControlLayerState = {
    id: getPrefixedId('control_layer'),
    type: 'control_layer',
    isEnabled: true,
    isLocked: false,
    opacity: 1,
    position: { x: 0, y: 0 },
    controlAdapter: {
      type: 't2i_adapter',
      model: zModelIdentifierField.parse(t2iAdapterModel),
      weight: typeof weight === 'number' ? weight : initialT2IAdapter.weight,
      beginEndStepPct,
    },
    name: null,
    objects: [],
    withTransparencyEffect: false,
  };

  return layer;
};

const parseIPAdapterToIPAdapterLayer: MetadataParseFunc<CanvasReferenceImageState> = async (metadataItem) => {
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
    begin_step_percent ?? initialIPAdapter.beginEndStepPct[0],
    end_step_percent ?? initialIPAdapter.beginEndStepPct[1],
  ];
  const imageDTO = image ? await getImageDTOSafe(image.image_name) : null;

  const layer: CanvasReferenceImageState = {
    id: getPrefixedId('ip_adapter'),
    type: 'reference_image',
    isEnabled: true,
    isLocked: false,
    name: null,
    ipAdapter: {
      type: 'ip_adapter',
      model: zModelIdentifierField.parse(ipAdapterModel),
      weight: typeof weight === 'number' ? weight : initialIPAdapter.weight,
      beginEndStepPct,
      clipVisionModel: initialIPAdapter.clipVisionModel, // TODO: This needs to be added to the zIPAdapterField...
      method: method ?? initialIPAdapter.method,
      image: imageDTO ? imageDTOToImageWithDims(imageDTO) : null,
    },
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
  guidance: parseGuidance,
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
  canvasV2Metadata: parseCanvasV2Metadata,
} as const;
