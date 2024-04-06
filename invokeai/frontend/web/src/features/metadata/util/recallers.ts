import { getStore } from 'app/store/nanostores/store';
import {
  controlAdapterRecalled,
  controlNetsReset,
  ipAdaptersReset,
  t2iAdaptersReset,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import { setHrfEnabled, setHrfMethod, setHrfStrength } from 'features/hrf/store/hrfSlice';
import type { LoRA } from 'features/lora/store/loraSlice';
import { loraRecalled, lorasReset } from 'features/lora/store/loraSlice';
import type {
  ControlNetConfigMetadata,
  IPAdapterConfigMetadata,
  MetadataRecallFunc,
  T2IAdapterConfigMetadata,
} from 'features/metadata/types';
import { modelSelected } from 'features/parameters/store/actions';
import {
  heightRecalled,
  initialImageChanged,
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
import type { ImageDTO } from 'services/api/types';

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

const recallInitialImage: MetadataRecallFunc<ImageDTO> = async (imageDTO) => {
  getStore().dispatch(initialImageChanged(imageDTO));
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

const recallModel: MetadataRecallFunc<ParameterModel> = (model) => {
  getStore().dispatch(modelSelected(model));
};

const recallRefinerModel: MetadataRecallFunc<ParameterSDXLRefinerModel> = (refinerModel) => {
  getStore().dispatch(refinerModelChanged(refinerModel));
};

const recallVAE: MetadataRecallFunc<ParameterVAEModel | null | undefined> = (vaeModel) => {
  if (!vaeModel) {
    getStore().dispatch(vaeSelected(null));
    return;
  }
  getStore().dispatch(vaeSelected(vaeModel));
};

const recallLoRA: MetadataRecallFunc<LoRA> = (lora) => {
  getStore().dispatch(loraRecalled(lora));
};

const recallAllLoRAs: MetadataRecallFunc<LoRA[]> = (loras) => {
  if (!loras.length) {
    return;
  }
  const { dispatch } = getStore();
  dispatch(lorasReset());
  loras.forEach((lora) => {
    dispatch(loraRecalled(lora));
  });
};

const recallControlNet: MetadataRecallFunc<ControlNetConfigMetadata> = (controlNet) => {
  getStore().dispatch(controlAdapterRecalled(controlNet));
};

const recallControlNets: MetadataRecallFunc<ControlNetConfigMetadata[]> = (controlNets) => {
  if (!controlNets.length) {
    return;
  }
  const { dispatch } = getStore();
  dispatch(controlNetsReset());
  controlNets.forEach((controlNet) => {
    dispatch(controlAdapterRecalled(controlNet));
  });
};

const recallT2IAdapter: MetadataRecallFunc<T2IAdapterConfigMetadata> = (t2iAdapter) => {
  getStore().dispatch(controlAdapterRecalled(t2iAdapter));
};

const recallT2IAdapters: MetadataRecallFunc<T2IAdapterConfigMetadata[]> = (t2iAdapters) => {
  if (!t2iAdapters.length) {
    return;
  }
  const { dispatch } = getStore();
  dispatch(t2iAdaptersReset());
  t2iAdapters.forEach((t2iAdapter) => {
    dispatch(controlAdapterRecalled(t2iAdapter));
  });
};

const recallIPAdapter: MetadataRecallFunc<IPAdapterConfigMetadata> = (ipAdapter) => {
  getStore().dispatch(controlAdapterRecalled(ipAdapter));
};

const recallIPAdapters: MetadataRecallFunc<IPAdapterConfigMetadata[]> = (ipAdapters) => {
  if (!ipAdapters.length) {
    return;
  }
  const { dispatch } = getStore();
  dispatch(ipAdaptersReset());
  ipAdapters.forEach((ipAdapter) => {
    dispatch(controlAdapterRecalled(ipAdapter));
  });
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
  initialImage: recallInitialImage,
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
