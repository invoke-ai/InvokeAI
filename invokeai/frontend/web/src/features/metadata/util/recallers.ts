import { getStore } from 'app/store/nanostores/store';
import { controlAdapterRecalled } from 'features/controlAdapters/store/controlAdaptersSlice';
import type { ControlNetConfig, IPAdapterConfig, T2IAdapterConfig } from 'features/controlAdapters/store/types';
import { setHrfEnabled, setHrfMethod, setHrfStrength } from 'features/hrf/store/hrfSlice';
import type { LoRA } from 'features/lora/store/loraSlice';
import { loraRecalled } from 'features/lora/store/loraSlice';
import type { MetadataRecallFunc } from 'features/metadata/types';
import { zModelIdentifierWithBase } from 'features/nodes/types/common';
import { modelSelected } from 'features/parameters/store/actions';
import {
  heightRecalled,
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
import type {
  ParameterCFGRescaleMultiplier,
  ParameterCFGScale,
  ParameterHeight,
  ParameterHRFEnabled,
  ParameterHRFMethod,
  ParameterNegativePrompt,
  ParameterNegativeStylePromptSDXL,
  ParameterPositivePrompt,
  ParameterPositiveStylePromptSDXL,
  ParameterScheduler,
  ParameterSDXLRefinerNegativeAestheticScore,
  ParameterSDXLRefinerPositiveAestheticScore,
  ParameterSDXLRefinerStart,
  ParameterSeed,
  ParameterSteps,
  ParameterStrength,
  ParameterWidth,
} from 'features/parameters/types/parameterSchemas';
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
import type { NonRefinerMainModelConfig, RefinerMainModelConfig, VAEModelConfig } from 'services/api/types';

const recallPositivePrompt: MetadataRecallFunc<ParameterPositivePrompt> = (positivePrompt) => {
  getStore().dispatch(setPositivePrompt(positivePrompt));
};

const recallNegativePrompt: MetadataRecallFunc<ParameterNegativePrompt> = (negativePrompt) => {
  getStore().dispatch(setNegativePrompt(negativePrompt));
};

const recallSDXLPositiveStylePrompt: MetadataRecallFunc<ParameterPositiveStylePromptSDXL> = (positiveStylePrompt) => {
  getStore().dispatch(setPositiveStylePromptSDXL(positiveStylePrompt));
};

const recallSDXLNegativeStylePrompt: MetadataRecallFunc<ParameterNegativeStylePromptSDXL> = (negativeStylePrompt) => {
  getStore().dispatch(setNegativeStylePromptSDXL(negativeStylePrompt));
};

const recallSeed: MetadataRecallFunc<ParameterSeed> = (seed) => {
  getStore().dispatch(setSeed(seed));
};

const recallCFGScale: MetadataRecallFunc<ParameterCFGScale> = (cfgScale) => {
  getStore().dispatch(setCfgScale(cfgScale));
};

const recallCFGRescaleMultiplier: MetadataRecallFunc<ParameterCFGRescaleMultiplier> = (cfgRescaleMultiplier) => {
  getStore().dispatch(setCfgRescaleMultiplier(cfgRescaleMultiplier));
};

const recallScheduler: MetadataRecallFunc<ParameterScheduler> = (scheduler) => {
  getStore().dispatch(setScheduler(scheduler));
};

const recallWidth: MetadataRecallFunc<ParameterWidth> = (width) => {
  getStore().dispatch(widthRecalled(width));
};

const recallHeight: MetadataRecallFunc<ParameterHeight> = (height) => {
  getStore().dispatch(heightRecalled(height));
};

const recallSteps: MetadataRecallFunc<ParameterSteps> = (steps) => {
  getStore().dispatch(setSteps(steps));
};

const recallStrength: MetadataRecallFunc<ParameterStrength> = (strength) => {
  getStore().dispatch(setImg2imgStrength(strength));
};

const recallHRFEnabled: MetadataRecallFunc<ParameterHRFEnabled> = (hrfEnabled) => {
  getStore().dispatch(setHrfEnabled(hrfEnabled));
};

const recallHRFStrength: MetadataRecallFunc<ParameterStrength> = (hrfStrength) => {
  getStore().dispatch(setHrfStrength(hrfStrength));
};

const recallHRFMethod: MetadataRecallFunc<ParameterHRFMethod> = (hrfMethod) => {
  getStore().dispatch(setHrfMethod(hrfMethod));
};

const recallRefinerSteps: MetadataRecallFunc<ParameterSteps> = (refinerSteps) => {
  getStore().dispatch(setRefinerSteps(refinerSteps));
};

const recallRefinerCFGScale: MetadataRecallFunc<ParameterCFGScale> = (refinerCFGScale) => {
  getStore().dispatch(setRefinerCFGScale(refinerCFGScale));
};

const recallRefinerScheduler: MetadataRecallFunc<ParameterScheduler> = (refinerScheduler) => {
  getStore().dispatch(setRefinerScheduler(refinerScheduler));
};

const recallRefinerPositiveAestheticScore: MetadataRecallFunc<ParameterSDXLRefinerPositiveAestheticScore> = (
  refinerPositiveAestheticScore
) => {
  getStore().dispatch(setRefinerPositiveAestheticScore(refinerPositiveAestheticScore));
};

const recallRefinerNegativeAestheticScore: MetadataRecallFunc<ParameterSDXLRefinerNegativeAestheticScore> = (
  refinerNegativeAestheticScore
) => {
  getStore().dispatch(setRefinerNegativeAestheticScore(refinerNegativeAestheticScore));
};

const recallRefinerStart: MetadataRecallFunc<ParameterSDXLRefinerStart> = (refinerStart) => {
  getStore().dispatch(setRefinerStart(refinerStart));
};

const recallModel: MetadataRecallFunc<NonRefinerMainModelConfig> = (model) => {
  const modelIdentifier = zModelIdentifierWithBase.parse(model);
  getStore().dispatch(modelSelected(modelIdentifier));
};

const recallRefinerModel: MetadataRecallFunc<RefinerMainModelConfig> = (refinerModel) => {
  const modelIdentifier = zModelIdentifierWithBase.parse(refinerModel);
  getStore().dispatch(refinerModelChanged(modelIdentifier));
};

const recallVAE: MetadataRecallFunc<VAEModelConfig | null | undefined> = (vaeModel) => {
  if (!vaeModel) {
    getStore().dispatch(vaeSelected(null));
    return;
  }
  const modelIdentifier = zModelIdentifierWithBase.parse(vaeModel);
  getStore().dispatch(vaeSelected(modelIdentifier));
};

const recallLoRA: MetadataRecallFunc<LoRA> = (lora) => {
  getStore().dispatch(loraRecalled(lora));
};

const recallAllLoRAs: MetadataRecallFunc<LoRA[]> = (loras) => {
  const { dispatch } = getStore();
  loras.forEach((lora) => {
    dispatch(loraRecalled(lora));
  });
};

const recallControlNet: MetadataRecallFunc<ControlNetConfig> = (controlNet) => {
  getStore().dispatch(controlAdapterRecalled(controlNet));
};

const recallControlNets: MetadataRecallFunc<ControlNetConfig[]> = (controlNets) => {
  const { dispatch } = getStore();
  controlNets.forEach((controlNet) => {
    dispatch(controlAdapterRecalled(controlNet));
  });
};

const recallT2IAdapter: MetadataRecallFunc<T2IAdapterConfig> = (t2iAdapter) => {
  getStore().dispatch(controlAdapterRecalled(t2iAdapter));
};

const recallT2IAdapters: MetadataRecallFunc<T2IAdapterConfig[]> = (t2iAdapters) => {
  const { dispatch } = getStore();
  t2iAdapters.forEach((t2iAdapter) => {
    dispatch(controlAdapterRecalled(t2iAdapter));
  });
};

const recallIPAdapter: MetadataRecallFunc<IPAdapterConfig> = (ipAdapter) => {
  getStore().dispatch(controlAdapterRecalled(ipAdapter));
};

const recallIPAdapters: MetadataRecallFunc<IPAdapterConfig[]> = (ipAdapters) => {
  const { dispatch } = getStore();
  ipAdapters.forEach((ipAdapter) => {
    dispatch(controlAdapterRecalled(ipAdapter));
  });
};

export const recallPrompts = (_metadata: unknown) => {
  // recallPositivePrompt(metadata);
  // recallNegativePrompt(metadata);
  // recallSDXLPositiveStylePrompt(metadata);
  // recallSDXLNegativeStylePrompt(metadata);
  // parameterNotSetToast(t('metadata.allPrompts'));
  // parameterSetToast(t('metadata.allPrompts'));
};

export const recallWidthAndHeight = (_metadata: unknown) => {
  // recallWidth(metadata);
  // recallHeight(metadata);
};

export const recallAll = async (_metadata: unknown) => {
  // if (!metadata) {
  //   allParameterNotSetToast();
  //   return;
  // }
  // // Update the main model first, as other parameters may depend on it.
  // await recallModel(metadata);
  // await Promise.allSettled([
  //   // Core parameters
  //   recallCFGScale(metadata),
  //   recallCFGRescaleMultiplier(metadata),
  //   recallPositivePrompt(metadata),
  //   recallNegativePrompt(metadata),
  //   recallScheduler(metadata),
  //   recallSeed(metadata),
  //   recallSteps(metadata),
  //   recallWidth(metadata),
  //   recallHeight(metadata),
  //   recallStrength(metadata),
  //   recallHRFEnabled(metadata),
  //   recallHRFMethod(metadata),
  //   recallHRFStrength(metadata),
  //   // SDXL parameters
  //   recallSDXLPositiveStylePrompt(metadata),
  //   recallSDXLNegativeStylePrompt(metadata),
  //   recallRefinerSteps(metadata),
  //   recallRefinerCFGScale(metadata),
  //   recallRefinerScheduler(metadata),
  //   recallRefinerPositiveAestheticScore(metadata),
  //   recallRefinerNegativeAestheticScore(metadata),
  //   recallRefinerStart(metadata),
  //   // Models
  //   recallVAE(metadata),
  //   recallRefinerModel(metadata),
  //   recallAllLoRAs(metadata),
  //   recallControlNets(metadata),
  //   recallT2IAdapters(metadata),
  //   recallIPAdapters(metadata),
  // ]);
  // allParameterSetToast();
};

export const recallers = {
  positivePrompt: recallPositivePrompt,
  negativePrompt: recallNegativePrompt,
  sdxlPositiveStylePrompt: recallSDXLPositiveStylePrompt,
  sdxlNegativeStylePrompt: recallSDXLNegativeStylePrompt,
  seed: recallSeed,
  cfgScale: recallCFGScale,
  cfgRescaleMultiplier: recallCFGRescaleMultiplier,
  scheduler: recallScheduler,
  width: recallWidth,
  height: recallHeight,
  steps: recallSteps,
  strength: recallStrength,
  hrfEnabled: recallHRFEnabled,
  hrfStrength: recallHRFStrength,
  hrfMethod: recallHRFMethod,
  refinerSteps: recallRefinerSteps,
  refinerCFGScale: recallRefinerCFGScale,
  refinerScheduler: recallRefinerScheduler,
  refinerPositiveAestheticScore: recallRefinerPositiveAestheticScore,
  refinerNegativeAestheticScore: recallRefinerNegativeAestheticScore,
  refinerStart: recallRefinerStart,
  model: recallModel,
  refinerModel: recallRefinerModel,
  vae: recallVAE,
  lora: recallLoRA,
  loras: recallAllLoRAs,
  controlNets: recallControlNets,
  controlNet: recallControlNet,
  t2iAdapters: recallT2IAdapters,
  t2iAdapter: recallT2IAdapter,
  ipAdapters: recallIPAdapters,
  ipAdapter: recallIPAdapter,
} as const;
