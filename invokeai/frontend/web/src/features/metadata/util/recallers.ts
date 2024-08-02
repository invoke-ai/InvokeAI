import { logger } from 'app/logging/logger';
import { getStore } from 'app/store/nanostores/store';
import { deepClone } from 'common/util/deepClone';
import {
  getBrushLineId,
  getCAId,
  getEraserLineId,
  getImageObjectId,
  getIPAId,
  getRectShapeId,
  getRGId,
} from 'features/controlLayers/konva/naming';
import {
  bboxHeightChanged,
  bboxWidthChanged,
  caRecalled,
  ipaRecalled,
  layerAllDeleted,
  layerRecalled,
  loraAllDeleted,
  loraRecalled,
  negativePrompt2Changed,
  negativePromptChanged,
  positivePrompt2Changed,
  positivePromptChanged,
  refinerModelChanged,
  rgRecalled,
  setCfgRescaleMultiplier,
  setCfgScale,
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
  vaeSelected,
} from 'features/controlLayers/store/canvasV2Slice';
import type {
  CanvasControlAdapterState,
  CanvasIPAdapterState,
  CanvasLayerState,
  LoRA,
  CanvasRegionalGuidanceState,
} from 'features/controlLayers/store/types';
import { setHrfEnabled, setHrfMethod, setHrfStrength } from 'features/hrf/store/hrfSlice';
import type {
  ControlNetConfigMetadata,
  IPAdapterConfigMetadata,
  MetadataRecallFunc,
  T2IAdapterConfigMetadata,
} from 'features/metadata/types';
import { fetchModelConfigByIdentifier } from 'features/metadata/util/modelFetchingHelpers';
import { modelSelected } from 'features/parameters/store/actions';
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
import { getImageDTO } from 'services/api/endpoints/images';
import { v4 as uuidv4 } from 'uuid';

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

const recallControlNet: MetadataRecallFunc<ControlNetConfigMetadata> = (controlNet) => {
  getStore().dispatch(controlAdapterRecalled(controlNet));
};

const recallControlNets: MetadataRecallFunc<ControlNetConfigMetadata[]> = (controlNets) => {
  const { dispatch } = getStore();
  dispatch(controlNetsReset());
  if (!controlNets.length) {
    return;
  }
  controlNets.forEach((controlNet) => {
    dispatch(controlAdapterRecalled(controlNet));
  });
};

const recallT2IAdapter: MetadataRecallFunc<T2IAdapterConfigMetadata> = (t2iAdapter) => {
  getStore().dispatch(controlAdapterRecalled(t2iAdapter));
};

const recallT2IAdapters: MetadataRecallFunc<T2IAdapterConfigMetadata[]> = (t2iAdapters) => {
  const { dispatch } = getStore();
  dispatch(t2iAdaptersReset());
  if (!t2iAdapters.length) {
    return;
  }
  t2iAdapters.forEach((t2iAdapter) => {
    dispatch(controlAdapterRecalled(t2iAdapter));
  });
};

const recallIPAdapter: MetadataRecallFunc<IPAdapterConfigMetadata> = (ipAdapter) => {
  getStore().dispatch(controlAdapterRecalled(ipAdapter));
};

const recallIPAdapters: MetadataRecallFunc<IPAdapterConfigMetadata[]> = (ipAdapters) => {
  const { dispatch } = getStore();
  dispatch(ipAdaptersReset());
  if (!ipAdapters.length) {
    return;
  }
  ipAdapters.forEach((ipAdapter) => {
    dispatch(controlAdapterRecalled(ipAdapter));
  });
};

const recallCA: MetadataRecallFunc<CanvasControlAdapterState> = async (ca) => {
  const { dispatch } = getStore();
  const clone = deepClone(ca);
  if (clone.image) {
    const imageDTO = await getImageDTO(clone.image.name);
    if (!imageDTO) {
      clone.image = null;
    }
  }
  if (clone.processedImage) {
    const imageDTO = await getImageDTO(clone.processedImage.name);
    if (!imageDTO) {
      clone.processedImage = null;
    }
  }
  if (clone.model) {
    try {
      await fetchModelConfigByIdentifier(clone.model);
    } catch {
      // MODEL SMITED!
      clone.model = null;
    }
  }
  // No clobber
  clone.id = getCAId(uuidv4());
  dispatch(caRecalled({ data: clone }));
  return;
};

const recallIPA: MetadataRecallFunc<CanvasIPAdapterState> = async (ipa) => {
  const { dispatch } = getStore();
  const clone = deepClone(ipa);
  if (clone.imageObject) {
    const imageDTO = await getImageDTO(clone.imageObject.name);
    if (!imageDTO) {
      clone.imageObject = null;
    }
  }
  if (clone.model) {
    try {
      await fetchModelConfigByIdentifier(clone.model);
    } catch {
      // MODEL SMITED!
      clone.model = null;
    }
  }
  // No clobber
  clone.id = getIPAId(uuidv4());
  dispatch(ipaRecalled({ data: clone }));
  return;
};

const recallRG: MetadataRecallFunc<CanvasRegionalGuidanceState> = async (rg) => {
  const { dispatch } = getStore();
  const clone = deepClone(rg);
  // Strip out the uploaded mask image property - this is an intermediate image
  clone.imageCache = null;

  for (const ipAdapter of clone.ipAdapters) {
    if (ipAdapter.imageObject) {
      const imageDTO = await getImageDTO(ipAdapter.imageObject.name);
      if (!imageDTO) {
        ipAdapter.imageObject = null;
      }
    }
    if (ipAdapter.model) {
      try {
        await fetchModelConfigByIdentifier(ipAdapter.model);
      } catch {
        // MODEL SMITED!
        ipAdapter.model = null;
      }
    }
    // No clobber
    ipAdapter.id = uuidv4();
  }
  clone.id = getRGId(uuidv4());
  dispatch(rgRecalled({ data: clone }));
  return;
};

//#region Control Layers
const recallLayer: MetadataRecallFunc<CanvasLayerState> = async (layer) => {
  const { dispatch } = getStore();
  const clone = deepClone(layer);
  const invalidObjects: string[] = [];
  for (const obj of clone.objects) {
    if (obj.type === 'image') {
      const imageDTO = await getImageDTO(obj.image.name);
      if (!imageDTO) {
        invalidObjects.push(obj.id);
      }
    }
  }
  clone.objects = clone.objects.filter(({ id }) => !invalidObjects.includes(id));
  for (const obj of clone.objects) {
    if (obj.type === 'brush_line') {
      obj.id = getBrushLineId(clone.id, uuidv4());
    } else if (obj.type === 'eraser_line') {
      obj.id = getEraserLineId(clone.id, uuidv4());
    } else if (obj.type === 'image') {
      obj.id = getImageObjectId(clone.id, uuidv4());
    } else if (obj.type === 'rect') {
      obj.id = getRectShapeId(clone.id, uuidv4());
    } else {
      logger('metadata').error(`Unknown object type ${obj.type}`);
    }
  }
  clone.id = getRGId(uuidv4());
  dispatch(layerRecalled({ data: clone }));
  return;
};

const recallLayers: MetadataRecallFunc<CanvasLayerState[]> = (layers) => {
  const { dispatch } = getStore();
  dispatch(layerAllDeleted());
  for (const l of layers) {
    recallLayer(l);
  }
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
  layer: recallLayer,
  layers: recallLayers,
} as const;
