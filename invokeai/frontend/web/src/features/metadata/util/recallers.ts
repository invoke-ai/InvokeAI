import { getStore } from 'app/store/nanostores/store';
import { bboxHeightChanged, bboxWidthChanged, canvasMetadataRecalled } from 'features/controlLayers/store/canvasSlice';
import { loraAllDeleted, loraRecalled } from 'features/controlLayers/store/lorasSlice';
import {
  negativePrompt2Changed,
  negativePromptChanged,
  positivePrompt2Changed,
  positivePromptChanged,
  refinerModelChanged,
  setCfgRescaleMultiplier,
  setCfgScale,
  setGuidance,
  setImg2imgStrength,
  setRefinerCFGScale,
  setRefinerNegativeAestheticScore,
  setRefinerPositiveAestheticScore,
  setRefinerScheduler,
  setRefinerStart,
  setRefinerSteps,
  setScheduler,
  setSeed,
  setSteps,
  t5EncoderModelSelected,
  vaeSelected,
} from 'features/controlLayers/store/paramsSlice';
import type { CanvasMetadata, LoRA } from 'features/controlLayers/store/types';
import { setHrfEnabled, setHrfMethod, setHrfStrength } from 'features/hrf/store/hrfSlice';
import type { MetadataRecallFunc } from 'features/metadata/types';
import { modelSelected } from 'features/parameters/store/actions';
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
  ParameterT5EncoderModel,
  ParameterVAEModel,
  ParameterWidth,
} from 'features/parameters/types/parameterSchemas';

const recallPositivePrompt: MetadataRecallFunc<ParameterPositivePrompt> = (positivePrompt) => {
  getStore().dispatch(positivePromptChanged(positivePrompt));
};

const recallNegativePrompt: MetadataRecallFunc<ParameterNegativePrompt> = (negativePrompt) => {
  getStore().dispatch(negativePromptChanged(negativePrompt));
};

const recallSDXLPositiveStylePrompt: MetadataRecallFunc<ParameterPositiveStylePromptSDXL> = (positiveStylePrompt) => {
  getStore().dispatch(positivePrompt2Changed(positiveStylePrompt));
};

const recallSDXLNegativeStylePrompt: MetadataRecallFunc<ParameterNegativeStylePromptSDXL> = (negativeStylePrompt) => {
  getStore().dispatch(negativePrompt2Changed(negativeStylePrompt));
};

const recallSeed: MetadataRecallFunc<ParameterSeed> = (seed) => {
  getStore().dispatch(setSeed(seed));
};

const recallCFGScale: MetadataRecallFunc<ParameterCFGScale> = (cfgScale) => {
  getStore().dispatch(setCfgScale(cfgScale));
};

const recallGuidance: MetadataRecallFunc<ParameterGuidance> = (guidance) => {
  getStore().dispatch(setGuidance(guidance));
};

const recallCFGRescaleMultiplier: MetadataRecallFunc<ParameterCFGRescaleMultiplier> = (cfgRescaleMultiplier) => {
  getStore().dispatch(setCfgRescaleMultiplier(cfgRescaleMultiplier));
};

const recallScheduler: MetadataRecallFunc<ParameterScheduler> = (scheduler) => {
  getStore().dispatch(setScheduler(scheduler));
};

const setSizeOptions = { updateAspectRatio: true, clamp: true };

const recallWidth: MetadataRecallFunc<ParameterWidth> = (width) => {
  getStore().dispatch(bboxWidthChanged({ width, ...setSizeOptions }));
};

const recallHeight: MetadataRecallFunc<ParameterHeight> = (height) => {
  getStore().dispatch(bboxHeightChanged({ height, ...setSizeOptions }));
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

const recallT5Encoder: MetadataRecallFunc<ParameterT5EncoderModel> = (t5EncoderModel) => {
  getStore().dispatch(t5EncoderModelSelected(t5EncoderModel));
};

const recallLoRA: MetadataRecallFunc<LoRA> = (lora) => {
  getStore().dispatch(loraRecalled({ lora }));
};

const recallAllLoRAs: MetadataRecallFunc<LoRA[]> = (loras) => {
  const { dispatch } = getStore();
  dispatch(loraAllDeleted());
  if (!loras.length) {
    return;
  }
  loras.forEach((lora) => {
    dispatch(loraRecalled({ lora }));
  });
};

const recallCanvasV2Metadata: MetadataRecallFunc<CanvasMetadata> = (canvasMetadata) => {
  if (
    canvasMetadata.controlLayers.length === 0 &&
    canvasMetadata.rasterLayers.length === 0 &&
    canvasMetadata.inpaintMasks.length === 0 &&
    canvasMetadata.referenceImages.length === 0 &&
    canvasMetadata.regionalGuidance.length === 0
  ) {
    // Nothing to recall
    return;
  }

  const { dispatch } = getStore();
  dispatch(canvasMetadataRecalled(canvasMetadata));
};

export const recallers = {
  positivePrompt: recallPositivePrompt,
  negativePrompt: recallNegativePrompt,
  sdxlPositiveStylePrompt: recallSDXLPositiveStylePrompt,
  sdxlNegativeStylePrompt: recallSDXLNegativeStylePrompt,
  seed: recallSeed,
  cfgScale: recallCFGScale,
  guidance: recallGuidance,
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
  t5EncoderModel: recallT5Encoder,
  canvasV2Metadata: recallCanvasV2Metadata,
} as const;
