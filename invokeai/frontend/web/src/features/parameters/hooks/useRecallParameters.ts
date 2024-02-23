import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { getStore } from 'app/store/nanostores/store';
import { useAppDispatch } from 'app/store/storeHooks';
import { toast } from 'common/util/toast';
import { CONTROLNET_PROCESSORS } from 'features/controlAdapters/store/constants';
import { controlAdapterRecalled, controlAdaptersReset } from 'features/controlAdapters/store/controlAdaptersSlice';
import type { ControlNetConfig, IPAdapterConfig, T2IAdapterConfig } from 'features/controlAdapters/store/types';
import {
  initialControlNet,
  initialIPAdapter,
  initialT2IAdapter,
} from 'features/controlAdapters/util/buildControlAdapter';
import { setHrfEnabled, setHrfMethod, setHrfStrength } from 'features/hrf/store/hrfSlice';
import type { LoRA } from 'features/lora/store/loraSlice';
import { defaultLoRAConfig, loraRecalled, lorasCleared } from 'features/lora/store/loraSlice';
import type { ModelIdentifierWithBase } from 'features/nodes/types/common';
import {
  zControlField,
  zIPAdapterField,
  zModelIdentifierWithBase,
  zT2IAdapterField,
} from 'features/nodes/types/common';
import { initialImageSelected, modelSelected } from 'features/parameters/store/actions';
import {
  heightRecalled,
  selectGenerationSlice,
  setCfgRescaleMultiplier,
  setCfgScale,
  setImg2imgStrength,
  setNegativePrompt,
  setPositivePrompt,
  setScheduler,
  setSeed,
  setSteps,
  vaeSelected,
  widthRecalled,
} from 'features/parameters/store/generationSlice';
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
import {
  fetchControlNetModel,
  fetchIPAdapterModel,
  fetchLoRAModel,
  fetchMainModelConfig,
  fetchRefinerModelConfig,
  fetchT2IAdapterModel,
  fetchVAEModelConfig,
  getModelKey,
  raiseIfBaseIncompatible,
} from 'features/parameters/util/modelFetchingHelpers';
import {
  refinerModelChanged,
  setNegativeStylePromptSDXL,
  setPositiveStylePromptSDXL,
  setRefinerCFGScale,
  setRefinerNegativeAestheticScore,
  setRefinerPositiveAestheticScore,
  setRefinerScheduler,
  setRefinerStart,
  setRefinerSteps,
} from 'features/sdxl/store/sdxlSlice';
import { t } from 'i18next';
import { get, isArray, isNil } from 'lodash-es';
import { useCallback } from 'react';
import type { BaseModelType, ImageDTO } from 'services/api/types';
import { v4 as uuidv4 } from 'uuid';

const selectModel = createMemoizedSelector(selectGenerationSlice, (generation) => generation.model);

/**
 * A function that recalls from metadata from the full metadata object.
 */
type MetadataRecallFunc = (metadata: unknown, withToast?: boolean) => void;

/**
 * A function that recalls metadata from a specific metadata item.
 */
type MetadataItemRecallFunc = (metadataItem: unknown, withToast?: boolean) => void;

/**
 * Raised when metadata recall fails.
 */
export class MetadataRecallError extends Error {
  /**
   * Create MetadataRecallError
   * @param {String} message
   */
  constructor(message: string) {
    super(message);
    this.name = this.constructor.name;
  }
}

export class InvalidMetadataPropertyType extends MetadataRecallError {}

const getProperty = <T = unknown>(
  obj: unknown,
  property: string,
  typeGuard: (val: unknown) => val is T = (val: unknown): val is T => true
): T => {
  const val = get(obj, property) as unknown;
  if (typeGuard(val)) {
    return val;
  }
  throw new InvalidMetadataPropertyType(`Property ${property} is not of expected type`);
};

const getCurrentBase = () => selectModel(getStore().getState())?.base;

const parameterSetToast = (parameter: string, description?: string) => {
  toast({
    title: t('toast.parameterSet', { parameter }),
    description,
    status: 'info',
    duration: 2500,
    isClosable: true,
  });
};

const parameterNotSetToast = (parameter: string, description?: string) => {
  toast({
    title: t('toast.parameterNotSet', { parameter }),
    description,
    status: 'warning',
    duration: 2500,
    isClosable: true,
  });
};

const allParameterSetToast = (description?: string) => {
  toast({
    title: t('toast.parametersSet'),
    status: 'info',
    description,
    duration: 2500,
    isClosable: true,
  });
};

const allParameterNotSetToast = (description?: string) => {
  toast({
    title: t('toast.parametersNotSet'),
    status: 'warning',
    description,
    duration: 2500,
    isClosable: true,
  });
};

const recall = (callback: () => void, parameter: string, withToast = true) => {
  try {
    callback();
    withToast && parameterSetToast(parameter);
  } catch (e) {
    withToast && parameterNotSetToast(parameter, (e as Error).message);
  }
};

const recallAsync = async (callback: () => Promise<void>, parameter: string, withToast = true) => {
  try {
    await callback();
    withToast && parameterSetToast(parameter);
  } catch (e) {
    withToast && parameterNotSetToast(parameter, (e as Error).message);
  }
};

export const recallPositivePrompt: MetadataRecallFunc = (metadata: unknown, withToast = true) => {
  recall(
    () => {
      const positive_prompt = getProperty(metadata, 'positive_prompt', isParameterPositivePrompt);
      getStore().dispatch(setPositivePrompt(positive_prompt));
    },
    t('metadata.positivePrompt'),
    withToast
  );
};

export const recallNegativePrompt: MetadataRecallFunc = (metadata: unknown, withToast = true) => {
  recall(
    () => {
      const negative_prompt = getProperty(metadata, 'negative_prompt', isParameterNegativePrompt);
      getStore().dispatch(setNegativePrompt(negative_prompt));
    },
    t('metadata.negativePrompt'),
    withToast
  );
};

export const recallSDXLPositiveStylePrompt: MetadataRecallFunc = (metadata: unknown, withToast = true) => {
  recall(
    () => {
      const positive_style_prompt = getProperty(metadata, 'positive_style_prompt', isParameterPositiveStylePromptSDXL);
      getStore().dispatch(setPositiveStylePromptSDXL(positive_style_prompt));
    },
    t('sdxl.posStylePrompt'),
    withToast
  );
};

export const recallSDXLNegativeStylePrompt: MetadataRecallFunc = (metadata: unknown, withToast = true) => {
  recall(
    () => {
      const negative_style_prompt = getProperty(metadata, 'negative_style_prompt', isParameterNegativeStylePromptSDXL);
      getStore().dispatch(setNegativeStylePromptSDXL(negative_style_prompt));
    },
    t('sdxl.negStylePrompt'),
    withToast
  );
};

export const recallSeed: MetadataRecallFunc = (metadata: unknown, withToast = true) => {
  recall(
    () => {
      const seed = getProperty(metadata, 'seed', isParameterSeed);
      getStore().dispatch(setSeed(seed));
    },
    t('metadata.seed'),
    withToast
  );
};

export const recallCFGScale: MetadataRecallFunc = (metadata: unknown, withToast = true) => {
  recall(
    () => {
      const cfg_scale = getProperty(metadata, 'cfg_scale', isParameterCFGScale);
      getStore().dispatch(setCfgScale(cfg_scale));
    },
    t('metadata.cfgScale'),
    withToast
  );
};

export const recallCFGRescaleMultiplier: MetadataRecallFunc = (metadata: unknown, withToast = true) => {
  recall(
    () => {
      const cfg_rescale_multiplier = getProperty(metadata, 'cfg_rescale_multiplier', isParameterCFGRescaleMultiplier);
      getStore().dispatch(setCfgRescaleMultiplier(cfg_rescale_multiplier));
    },
    t('metadata.cfgRescaleMultiplier'),
    withToast
  );
};

export const recallScheduler: MetadataRecallFunc = (metadata: unknown, withToast = true) => {
  recall(
    () => {
      const scheduler = getProperty(metadata, 'scheduler', isParameterScheduler);
      getStore().dispatch(setScheduler(scheduler));
    },
    t('metadata.scheduler'),
    withToast
  );
};

export const recallWidth: MetadataRecallFunc = (metadata: unknown, withToast = true) => {
  recall(
    () => {
      const width = getProperty(metadata, 'width', isParameterWidth);
      getStore().dispatch(widthRecalled(width));
    },
    t('metadata.width'),
    withToast
  );
};

export const recallHeight: MetadataRecallFunc = (metadata: unknown, withToast = true) => {
  recall(
    () => {
      const height = getProperty(metadata, 'height', isParameterHeight);
      getStore().dispatch(heightRecalled(height));
    },
    t('metadata.height'),
    withToast
  );
};

export const recallSteps: MetadataRecallFunc = (metadata: unknown, withToast = true) => {
  recall(
    () => {
      const steps = getProperty(metadata, 'steps', isParameterSteps);
      getStore().dispatch(setSteps(steps));
    },
    t('metadata.steps'),
    withToast
  );
};

export const recallStrength: MetadataRecallFunc = (metadata: unknown, withToast = true) => {
  recall(
    () => {
      const strength = getProperty(metadata, 'strength', isParameterStrength);
      getStore().dispatch(setImg2imgStrength(strength));
    },
    t('metadata.strength'),
    withToast
  );
};

export const recallHRFEnabled: MetadataRecallFunc = (metadata: unknown, withToast = true) => {
  recall(
    () => {
      const hrf_enabled = getProperty(metadata, 'hrf_enabled', isParameterHRFEnabled);
      getStore().dispatch(setHrfEnabled(hrf_enabled));
    },
    t('hrf.metadata.enabled'),
    withToast
  );
};

export const recallHRFStrength: MetadataRecallFunc = (metadata: unknown, withToast = true) => {
  recall(
    () => {
      const hrf_strength = getProperty(metadata, 'hrf_strength', isParameterStrength);
      getStore().dispatch(setHrfStrength(hrf_strength));
    },
    t('hrf.metadata.strength'),
    withToast
  );
};

export const recallHRFMethod: MetadataRecallFunc = (metadata: unknown, withToast = true) => {
  recall(
    () => {
      const hrf_method = getProperty(metadata, 'hrf_method', isParameterHRFMethod);
      getStore().dispatch(setHrfMethod(hrf_method));
    },
    t('hrf.metadata.method'),
    withToast
  );
};

export const recallRefinerSteps: MetadataRecallFunc = (metadata: unknown, withToast = true) => {
  recall(
    () => {
      const refiner_steps = getProperty(metadata, 'refiner_steps', isParameterSteps);
      getStore().dispatch(setRefinerSteps(refiner_steps));
    },
    t('sdxl.steps'),
    withToast
  );
};

export const recallRefinerCFGScale: MetadataRecallFunc = (metadata: unknown, withToast = true) => {
  recall(
    () => {
      const refiner_cfg_scale = getProperty(metadata, 'refiner_cfg_scale', isParameterCFGScale);
      getStore().dispatch(setRefinerCFGScale(refiner_cfg_scale));
    },
    t('sdxl.cfgScale'),
    withToast
  );
};

export const recallRefinerScheduler: MetadataRecallFunc = (metadata: unknown, withToast = true) => {
  recall(
    () => {
      const refiner_scheduler = getProperty(metadata, 'refiner_scheduler', isParameterScheduler);
      getStore().dispatch(setRefinerScheduler(refiner_scheduler));
    },
    t('sdxl.cfgScale'),
    withToast
  );
};

export const recallRefinerPositiveAestheticScore: MetadataRecallFunc = (metadata: unknown, withToast = true) => {
  recall(
    () => {
      const refiner_positive_aesthetic_score = getProperty(
        metadata,
        'refiner_positive_aesthetic_score',
        isParameterSDXLRefinerPositiveAestheticScore
      );
      getStore().dispatch(setRefinerPositiveAestheticScore(refiner_positive_aesthetic_score));
    },
    t('sdxl.posAestheticScore'),
    withToast
  );
};

export const recallRefinerNegativeAestheticScore: MetadataRecallFunc = (metadata: unknown, withToast = true) => {
  recall(
    () => {
      const refiner_negative_aesthetic_score = getProperty(
        metadata,
        'refiner_negative_aesthetic_score',
        isParameterSDXLRefinerNegativeAestheticScore
      );
      getStore().dispatch(setRefinerNegativeAestheticScore(refiner_negative_aesthetic_score));
    },
    t('sdxl.negAestheticScore'),
    withToast
  );
};

export const recallRefinerStart: MetadataRecallFunc = (metadata: unknown, withToast = true) => {
  recall(
    () => {
      const refiner_start = getProperty(metadata, 'refiner_start', isParameterSDXLRefinerStart);
      getStore().dispatch(setRefinerStart(refiner_start));
    },
    t('sdxl.refinerStart'),
    withToast
  );
};

export const prepareMainModelMetadataItem = async (model: unknown): Promise<ModelIdentifierWithBase> => {
  const key = await getModelKey(model, 'main');
  const mainModel = await fetchMainModelConfig(key);
  return zModelIdentifierWithBase.parse(mainModel);
};

const recallModelAsync: MetadataRecallFunc = async (metadata: unknown, withToast = true) => {
  await recallAsync(
    async () => {
      const modelMetadataItem = getProperty(metadata, 'model');
      const model = await prepareMainModelMetadataItem(modelMetadataItem);
      getStore().dispatch(modelSelected(model));
    },
    t('metadata.model'),
    withToast
  );
};

export const prepareRefinerMetadataItem = async (
  model: unknown,
  currentBase: BaseModelType | undefined
): Promise<ModelIdentifierWithBase> => {
  const key = await getModelKey(model, 'main');
  const refinerModel = await fetchRefinerModelConfig(key);
  raiseIfBaseIncompatible('sdxl-refiner', currentBase, 'Refiner incompatible with currently-selected model');
  return zModelIdentifierWithBase.parse(refinerModel);
};

const recallRefinerModelAsync: MetadataRecallFunc = async (metadata: unknown, withToast = true) => {
  await recallAsync(
    async () => {
      const refinerMetadataItem = getProperty(metadata, 'refiner_model');
      const refiner = await prepareRefinerMetadataItem(refinerMetadataItem, getCurrentBase());
      getStore().dispatch(refinerModelChanged(refiner));
    },
    t('sdxl.refinerModel'),
    withToast
  );
};

export const prepareVAEMetadataItem = async (
  vae: unknown,
  currentBase: BaseModelType | undefined
): Promise<ModelIdentifierWithBase> => {
  const key = await getModelKey(vae, 'vae');
  const vaeModel = await fetchVAEModelConfig(key);
  raiseIfBaseIncompatible(vaeModel.base, currentBase, 'VAE incompatible with currently-selected model');
  return zModelIdentifierWithBase.parse(vaeModel);
};

const recallVAEAsync: MetadataRecallFunc = async (metadata: unknown, withToast = true) => {
  await recallAsync(
    async () => {
      const currentBase = getCurrentBase();
      const vaeMetadataItem = getProperty(metadata, 'vae');
      if (isNil(vaeMetadataItem)) {
        getStore().dispatch(vaeSelected(null));
      } else {
        const vae = await prepareVAEMetadataItem(vaeMetadataItem, currentBase);
        getStore().dispatch(vaeSelected(vae));
      }
    },
    t('metadata.vae'),
    withToast
  );
};

export const prepareLoRAMetadataItem = async (
  loraMetadataItem: unknown,
  currentBase: BaseModelType | undefined
): Promise<LoRA> => {
  const lora = getProperty(loraMetadataItem, 'lora');
  const weight = getProperty(loraMetadataItem, 'weight');
  const key = await getModelKey(lora, 'lora');
  const loraModel = await fetchLoRAModel(key);
  raiseIfBaseIncompatible(loraModel.base, currentBase, 'LoRA incompatible with currently-selected model');
  return {
    key: loraModel.key,
    base: loraModel.base,
    weight: isParameterLoRAWeight(weight) ? weight : defaultLoRAConfig.weight,
    isEnabled: true,
  };
};

const recallLoRAAsync: MetadataItemRecallFunc = async (metadataItem: unknown, withToast = true) => {
  await recallAsync(
    async () => {
      const currentBase = getCurrentBase();
      const lora = await prepareLoRAMetadataItem(metadataItem, currentBase);
      getStore().dispatch(loraRecalled(lora));
    },
    t('models.lora'),
    withToast
  );
};

export const prepareControlNetMetadataItem = async (
  metadataItem: unknown,
  currentBase: BaseModelType | undefined
): Promise<ControlNetConfig> => {
  const control_model = getProperty(metadataItem, 'control_model');
  const key = await getModelKey(control_model, 'controlnet');
  const controlNetModel = await fetchControlNetModel(key);
  raiseIfBaseIncompatible(controlNetModel.base, currentBase, 'ControlNet incompatible with currently-selected model');

  const image = zControlField.shape.image.nullish().catch(null).parse(getProperty(metadataItem, 'image'));
  const control_weight = zControlField.shape.control_weight
    .nullish()
    .catch(null)
    .parse(getProperty(metadataItem, 'control_weight'));
  const begin_step_percent = zControlField.shape.begin_step_percent
    .nullish()
    .catch(null)
    .parse(getProperty(metadataItem, 'begin_step_percent'));
  const end_step_percent = zControlField.shape.end_step_percent
    .nullish()
    .catch(null)
    .parse(getProperty(metadataItem, 'end_step_percent'));
  const control_mode = zControlField.shape.control_mode
    .nullish()
    .catch(null)
    .parse(getProperty(metadataItem, 'control_mode'));
  const resize_mode = zControlField.shape.resize_mode
    .nullish()
    .catch(null)
    .parse(getProperty(metadataItem, 'resize_mode'));

  // We don't save the original image that was processed into a control image, only the processed image
  const processorType = 'none';
  const processorNode = CONTROLNET_PROCESSORS.none.default;

  const controlnet: ControlNetConfig = {
    type: 'controlnet',
    isEnabled: true,
    model: zModelIdentifierWithBase.parse(controlNetModel),
    weight: typeof control_weight === 'number' ? control_weight : initialControlNet.weight,
    beginStepPct: begin_step_percent ?? initialControlNet.beginStepPct,
    endStepPct: end_step_percent ?? initialControlNet.endStepPct,
    controlMode: control_mode ?? initialControlNet.controlMode,
    resizeMode: resize_mode ?? initialControlNet.resizeMode,
    controlImage: image?.image_name ?? null,
    processedControlImage: image?.image_name ?? null,
    processorType,
    processorNode,
    shouldAutoConfig: true,
    id: uuidv4(),
  };

  return controlnet;
};

const recallControlNetAsync: MetadataItemRecallFunc = async (metadataItem: unknown, withToast = true) => {
  await recallAsync(
    async () => {
      const currentBase = getCurrentBase();
      const controlNetConfig = await prepareControlNetMetadataItem(metadataItem, currentBase);
      getStore().dispatch(controlAdapterRecalled(controlNetConfig));
    },
    t('common.controlNet'),
    withToast
  );
};

export const prepareT2IAdapterMetadataItem = async (
  metadataItem: unknown,
  currentBase: BaseModelType | undefined
): Promise<T2IAdapterConfig> => {
  const t2i_adapter_model = getProperty(metadataItem, 't2i_adapter_model');
  const key = await getModelKey(t2i_adapter_model, 't2i_adapter');
  const t2iAdapterModel = await fetchT2IAdapterModel(key);
  raiseIfBaseIncompatible(t2iAdapterModel.base, currentBase, 'T2I Adapter incompatible with currently-selected model');

  const image = zT2IAdapterField.shape.image.nullish().catch(null).parse(getProperty(metadataItem, 'image'));
  const weight = zT2IAdapterField.shape.weight.nullish().catch(null).parse(getProperty(metadataItem, 'weight'));
  const begin_step_percent = zT2IAdapterField.shape.begin_step_percent
    .nullish()
    .catch(null)
    .parse(getProperty(metadataItem, 'begin_step_percent'));
  const end_step_percent = zT2IAdapterField.shape.end_step_percent
    .nullish()
    .catch(null)
    .parse(getProperty(metadataItem, 'end_step_percent'));
  const resize_mode = zT2IAdapterField.shape.resize_mode
    .nullish()
    .catch(null)
    .parse(getProperty(metadataItem, 'resize_mode'));

  // We don't save the original image that was processed into a control image, only the processed image
  const processorType = 'none';
  const processorNode = CONTROLNET_PROCESSORS.none.default;

  const t2iAdapter: T2IAdapterConfig = {
    type: 't2i_adapter',
    isEnabled: true,
    model: zModelIdentifierWithBase.parse(t2iAdapterModel),
    weight: typeof weight === 'number' ? weight : initialT2IAdapter.weight,
    beginStepPct: begin_step_percent ?? initialT2IAdapter.beginStepPct,
    endStepPct: end_step_percent ?? initialT2IAdapter.endStepPct,
    resizeMode: resize_mode ?? initialT2IAdapter.resizeMode,
    controlImage: image?.image_name ?? null,
    processedControlImage: image?.image_name ?? null,
    processorType,
    processorNode,
    shouldAutoConfig: true,
    id: uuidv4(),
  };

  return t2iAdapter;
};

const recallT2IAdapterAsync: MetadataItemRecallFunc = async (metadataItem: unknown, withToast = true) => {
  await recallAsync(
    async () => {
      const currentBase = getCurrentBase();
      const t2iAdapterConfig = await prepareT2IAdapterMetadataItem(metadataItem, currentBase);
      getStore().dispatch(controlAdapterRecalled(t2iAdapterConfig));
    },
    t('common.t2iAdapter'),
    withToast
  );
};

export const prepareIPAdapterMetadataItem = async (
  metadataItem: unknown,
  currentBase: BaseModelType | undefined
): Promise<IPAdapterConfig> => {
  const ip_adapter_model = getProperty(metadataItem, 'ip_adapter_model');
  const key = await getModelKey(ip_adapter_model, 'ip_adapter');
  const ipAdapterModel = await fetchIPAdapterModel(key);
  raiseIfBaseIncompatible(ipAdapterModel.base, currentBase, 'T2I Adapter incompatible with currently-selected model');

  const image = zIPAdapterField.shape.image.nullish().catch(null).parse(getProperty(metadataItem, 'image'));
  const weight = zIPAdapterField.shape.weight.nullish().catch(null).parse(getProperty(metadataItem, 'weight'));
  const begin_step_percent = zIPAdapterField.shape.begin_step_percent
    .nullish()
    .catch(null)
    .parse(getProperty(metadataItem, 'begin_step_percent'));
  const end_step_percent = zIPAdapterField.shape.end_step_percent
    .nullish()
    .catch(null)
    .parse(getProperty(metadataItem, 'end_step_percent'));

  const ipAdapter: IPAdapterConfig = {
    id: uuidv4(),
    type: 'ip_adapter',
    isEnabled: true,
    model: zModelIdentifierWithBase.parse(ipAdapterModel),
    controlImage: image?.image_name ?? null,
    weight: weight ?? initialIPAdapter.weight,
    beginStepPct: begin_step_percent ?? initialIPAdapter.beginStepPct,
    endStepPct: end_step_percent ?? initialIPAdapter.endStepPct,
  };

  return ipAdapter;
};

const recallIPAdapterAsync: MetadataItemRecallFunc = async (metadataItem: unknown, withToast) => {
  await recallAsync(
    async () => {
      const currentBase = getCurrentBase();
      const ipAdapterConfig = await prepareIPAdapterMetadataItem(metadataItem, currentBase);
      getStore().dispatch(controlAdapterRecalled(ipAdapterConfig));
    },
    t('common.ipAdapter'),
    withToast
  );
};

export const useRecallParameters = () => {
  const dispatch = useAppDispatch();

  const recallBothPrompts = useCallback<MetadataRecallFunc>(
    (metadata: unknown) => {
      const positive_prompt = getProperty(metadata, 'positive_prompt');
      const negative_prompt = getProperty(metadata, 'negative_prompt');
      const positive_style_prompt = getProperty(metadata, 'positive_style_prompt');
      const negative_style_prompt = getProperty(metadata, 'negative_style_prompt');
      if (
        isParameterPositivePrompt(positive_prompt) ||
        isParameterNegativePrompt(negative_prompt) ||
        isParameterPositiveStylePromptSDXL(positive_style_prompt) ||
        isParameterNegativeStylePromptSDXL(negative_style_prompt)
      ) {
        if (isParameterPositivePrompt(positive_prompt)) {
          dispatch(setPositivePrompt(positive_prompt));
        }

        if (isParameterNegativePrompt(negative_prompt)) {
          dispatch(setNegativePrompt(negative_prompt));
        }

        if (isParameterPositiveStylePromptSDXL(positive_style_prompt)) {
          dispatch(setPositiveStylePromptSDXL(positive_style_prompt));
        }

        if (isParameterPositiveStylePromptSDXL(negative_style_prompt)) {
          dispatch(setNegativeStylePromptSDXL(negative_style_prompt));
        }

        parameterSetToast(t('metadata.allPrompts'));
        return;
      }
      parameterNotSetToast(t('metadata.allPrompts'));
    },
    [dispatch]
  );

  const recallWidthAndHeight = useCallback<MetadataRecallFunc>(
    (metadata: unknown) => {
      const width = getProperty(metadata, 'width');
      const height = getProperty(metadata, 'height');

      if (!isParameterWidth(width)) {
        allParameterNotSetToast();
        return;
      }
      if (!isParameterHeight(height)) {
        allParameterNotSetToast();
        return;
      }
      dispatch(heightRecalled(height));
      dispatch(widthRecalled(width));
      allParameterSetToast();
    },
    [dispatch]
  );

  const sendToImageToImage = useCallback(
    (image: ImageDTO) => {
      dispatch(initialImageSelected(image));
    },
    [dispatch]
  );

  const recallAllParameters = useCallback<MetadataRecallFunc>(
    async (metadata: unknown) => {
      if (!metadata) {
        allParameterNotSetToast();
        return;
      }

      await recallModelAsync(metadata, false);

      recallCFGScale(metadata, false);
      recallCFGRescaleMultiplier(metadata, false);
      recallPositivePrompt(metadata, false);
      recallNegativePrompt(metadata, false);
      recallScheduler(metadata, false);
      recallSeed(metadata, false);
      recallSteps(metadata, false);
      recallWidth(metadata, false);
      recallHeight(metadata, false);
      recallStrength(metadata, false);
      recallHRFEnabled(metadata, false);
      recallHRFMethod(metadata, false);
      recallHRFStrength(metadata, false);

      // SDXL
      recallSDXLPositiveStylePrompt(metadata, false);
      recallSDXLNegativeStylePrompt(metadata, false);
      recallRefinerSteps(metadata, false);
      recallRefinerCFGScale(metadata, false);
      recallRefinerScheduler(metadata, false);
      recallRefinerPositiveAestheticScore(metadata, false);
      recallRefinerNegativeAestheticScore(metadata, false);
      recallRefinerStart(metadata, false);

      await recallVAEAsync(metadata, false);
      await recallRefinerModelAsync(metadata, false);

      dispatch(lorasCleared());
      const loraMetadataArray = getProperty(metadata, 'loras');
      if (isArray(loraMetadataArray)) {
        loraMetadataArray.forEach(async (loraMetadataItem) => {
          await recallLoRAAsync(loraMetadataItem, false);
        });
      }

      dispatch(controlAdaptersReset());
      const controlNetMetadataArray = getProperty(metadata, 'controlnets');
      if (isArray(controlNetMetadataArray)) {
        controlNetMetadataArray.forEach(async (controlNetMetadataItem) => {
          await recallControlNetAsync(controlNetMetadataItem, false);
        });
      }

      const ipAdapterMetadataArray = getProperty(metadata, 'ipAdapters');
      if (isArray(ipAdapterMetadataArray)) {
        ipAdapterMetadataArray.forEach(async (ipAdapterMetadataItem) => {
          await recallIPAdapterAsync(ipAdapterMetadataItem, false);
        });
      }

      const t2iAdapterMetadataArray = getProperty(metadata, 't2iAdapters');
      if (isArray(t2iAdapterMetadataArray)) {
        t2iAdapterMetadataArray.forEach(async (t2iAdapterMetadataItem) => {
          await recallT2IAdapterAsync(t2iAdapterMetadataItem, false);
        });
      }

      allParameterSetToast();
    },
    [dispatch]
  );

  return {
    recallBothPrompts,
    recallPositivePrompt,
    recallNegativePrompt,
    recallSDXLPositiveStylePrompt,
    recallSDXLNegativeStylePrompt,
    recallSeed,
    recallCFGScale,
    recallCFGRescaleMultiplier,
    recallModel: recallModelAsync,
    recallScheduler,
    recallVaeModel: recallVAEAsync,
    recallSteps,
    recallWidth,
    recallHeight,
    recallWidthAndHeight,
    recallStrength,
    recallHRFEnabled,
    recallHRFStrength,
    recallHRFMethod,
    recallLoRA: recallLoRAAsync,
    recallControlNet: recallControlNetAsync,
    recallIPAdapter: recallIPAdapterAsync,
    recallT2IAdapter: recallT2IAdapterAsync,
    recallAllParameters,
    recallRefinerModel: recallRefinerModelAsync,
    sendToImageToImage,
  };
};
