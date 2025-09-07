import { deepClone } from 'common/util/deepClone';
import { merge } from 'es-toolkit/compat';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type {
  CanvasControlLayerState,
  CanvasImageState,
  CanvasInpaintMaskState,
  CanvasRasterLayerState,
  CanvasRegionalGuidanceState,
  ChatGPT4oReferenceImageConfig,
  ControlLoRAConfig,
  ControlNetConfig,
  FluxKontextReferenceImageConfig,
  FLUXReduxConfig,
  Gemini2_5ReferenceImageConfig,
  ImageWithDims,
  IPAdapterConfig,
  RefImageState,
  RgbColor,
  T2IAdapterConfig,
} from 'features/controlLayers/store/types';
import type { ImageField } from 'features/nodes/types/common';
import type { ImageDTO } from 'services/api/types';
import { assert } from 'tsafe';
import type { PartialDeep } from 'type-fest';

export const imageDTOToImageObject = (imageDTO: ImageDTO, overrides?: Partial<CanvasImageState>): CanvasImageState => {
  const { width, height, image_name } = imageDTO;
  return {
    id: getPrefixedId('image'),
    type: 'image',
    image: {
      image_name,
      width,
      height,
    },
    ...overrides,
  };
};

export const imageDTOToImageWithDims = ({ image_name, width, height }: ImageDTO): ImageWithDims => ({
  image_name,
  width,
  height,
});

export const imageDTOToImageField = ({ image_name }: ImageDTO): ImageField => ({ image_name });

const DEFAULT_RG_MASK_FILL_COLORS: RgbColor[] = [
  { r: 121, g: 157, b: 219 }, // rgb(121, 157, 219)
  { r: 131, g: 214, b: 131 }, // rgb(131, 214, 131)
  { r: 250, g: 225, b: 80 }, // rgb(250, 225, 80)
  { r: 220, g: 144, b: 101 }, // rgb(220, 144, 101)
  { r: 224, g: 117, b: 117 }, // rgb(224, 117, 117)
  { r: 213, g: 139, b: 202 }, // rgb(213, 139, 202)
  { r: 161, g: 120, b: 214 }, // rgb(161, 120, 214)
];
const buildMaskFillCycler = (initialIndex: number): (() => RgbColor) => {
  let lastFillIndex = initialIndex;

  return () => {
    lastFillIndex = (lastFillIndex + 1) % DEFAULT_RG_MASK_FILL_COLORS.length;
    const fill = DEFAULT_RG_MASK_FILL_COLORS[lastFillIndex];
    assert(fill, 'This should never happen');
    return fill;
  };
};

const getInpaintMaskFillColor = buildMaskFillCycler(3);
const getRegionalGuidanceMaskFillColor = buildMaskFillCycler(0);

export const initialIPAdapter: IPAdapterConfig = {
  type: 'ip_adapter',
  image: null,
  model: null,
  beginEndStepPct: [0, 1],
  method: 'full',
  clipVisionModel: 'ViT-H',
  weight: 1,
};
export const initialFLUXRedux: FLUXReduxConfig = {
  type: 'flux_redux',
  image: null,
  model: null,
  imageInfluence: 'highest',
};
export const initialChatGPT4oReferenceImage: ChatGPT4oReferenceImageConfig = {
  type: 'chatgpt_4o_reference_image',
  image: null,
  model: null,
};
export const initialGemini2_5ReferenceImage: Gemini2_5ReferenceImageConfig = {
  type: 'gemini_2_5_reference_image',
  image: null,
  model: null,
};
export const initialFluxKontextReferenceImage: FluxKontextReferenceImageConfig = {
  type: 'flux_kontext_reference_image',
  image: null,
  model: null,
};
export const initialT2IAdapter: T2IAdapterConfig = {
  type: 't2i_adapter',
  model: null,
  weight: 1,
  beginEndStepPct: [0, 1],
};
export const initialControlNet: ControlNetConfig = {
  type: 'controlnet',
  model: null,
  weight: 0.75,
  beginEndStepPct: [0, 0.75],
  controlMode: 'balanced',
};
export const initialControlLoRA: ControlLoRAConfig = {
  type: 'control_lora',
  model: null,
  weight: 0.75,
};

export type RasterLayerAdjustments = NonNullable<CanvasRasterLayerState['adjustments']>;

export const makeDefaultRasterLayerAdjustments = (mode: 'simple' | 'curves' = 'simple'): RasterLayerAdjustments => ({
  version: 1,
  enabled: true,
  collapsed: false,
  mode,
  simple: { brightness: 0, contrast: 0, saturation: 0, temperature: 0, tint: 0, sharpness: 0 },
  curves: {
    master: [
      [0, 0],
      [255, 255],
    ],
    r: [
      [0, 0],
      [255, 255],
    ],
    g: [
      [0, 0],
      [255, 255],
    ],
    b: [
      [0, 0],
      [255, 255],
    ],
  },
});

export const getReferenceImageState = (id: string, overrides?: PartialDeep<RefImageState>): RefImageState => {
  const entityState: RefImageState = {
    id,
    isEnabled: true,
    config: deepClone(initialIPAdapter),
  };
  merge(entityState, overrides);
  return entityState;
};

export const getRegionalGuidanceState = (
  id: string,
  overrides?: PartialDeep<CanvasRegionalGuidanceState>
): CanvasRegionalGuidanceState => {
  const entityState: CanvasRegionalGuidanceState = {
    id,
    name: null,
    isLocked: false,
    type: 'regional_guidance',
    isEnabled: true,
    objects: [],
    fill: {
      style: 'solid',
      color: getRegionalGuidanceMaskFillColor(),
    },
    opacity: 0.5,
    position: { x: 0, y: 0 },
    autoNegative: false,
    positivePrompt: null,
    negativePrompt: null,
    referenceImages: [],
  };
  merge(entityState, overrides);
  return entityState;
};

export const getControlLayerState = (
  id: string,
  overrides?: PartialDeep<CanvasControlLayerState>
): CanvasControlLayerState => {
  const entityState: CanvasControlLayerState = {
    id,
    name: null,
    type: 'control_layer',
    isEnabled: true,
    isLocked: false,
    withTransparencyEffect: true,
    objects: [],
    opacity: 1,
    position: { x: 0, y: 0 },
    controlAdapter: deepClone(initialControlNet),
  };
  merge(entityState, overrides);
  return entityState;
};

export const getRasterLayerState = (
  id: string,
  overrides?: PartialDeep<CanvasRasterLayerState>
): CanvasRasterLayerState => {
  const entityState: CanvasRasterLayerState = {
    id,
    name: null,
    type: 'raster_layer',
    isEnabled: true,
    isLocked: false,
    objects: [],
    opacity: 1,
    position: { x: 0, y: 0 },
    adjustments: undefined,
  };
  merge(entityState, overrides);
  return entityState;
};

export const getInpaintMaskState = (
  id: string,
  overrides?: PartialDeep<CanvasInpaintMaskState>
): CanvasInpaintMaskState => {
  const entityState: CanvasInpaintMaskState = {
    id,
    name: null,
    type: 'inpaint_mask',
    isEnabled: true,
    isLocked: false,
    objects: [],
    opacity: 1,
    position: { x: 0, y: 0 },
    fill: {
      style: 'diagonal',
      color: getInpaintMaskFillColor(),
    },
    noiseLevel: undefined,
    denoiseLimit: undefined,
  };
  merge(entityState, overrides);
  return entityState;
};

const convertRasterLayerToControlLayer = (
  newId: string,
  rasterLayerState: CanvasRasterLayerState,
  overrides?: PartialDeep<CanvasControlLayerState>
): CanvasControlLayerState => {
  const { name, objects, position } = rasterLayerState;
  const controlLayerState = getControlLayerState(newId, {
    name,
    objects,
    position,
  });
  merge(controlLayerState, overrides);
  return controlLayerState;
};

const convertRasterLayerToInpaintMask = (
  newId: string,
  rasterLayerState: CanvasRasterLayerState,
  overrides?: PartialDeep<CanvasInpaintMaskState>
): CanvasInpaintMaskState => {
  const { name, objects, position } = rasterLayerState;
  const inpaintMaskState = getInpaintMaskState(newId, {
    name,
    objects,
    position,
  });
  merge(inpaintMaskState, overrides);
  return inpaintMaskState;
};

const convertRasterLayerToRegionalGuidance = (
  newId: string,
  rasterLayerState: CanvasRasterLayerState,
  overrides?: PartialDeep<CanvasRegionalGuidanceState>
): CanvasRegionalGuidanceState => {
  const { name, objects, position } = rasterLayerState;
  const regionalGuidanceState = getRegionalGuidanceState(newId, {
    name,
    objects,
    position,
  });
  merge(regionalGuidanceState, overrides);
  return regionalGuidanceState;
};

const convertControlLayerToRasterLayer = (
  newId: string,
  controlLayerState: CanvasControlLayerState,
  overrides?: PartialDeep<CanvasRasterLayerState>
): CanvasRasterLayerState => {
  const { name, objects, position } = controlLayerState;
  const rasterLayerState = getRasterLayerState(newId, {
    name,
    objects,
    position,
  });
  merge(rasterLayerState, overrides);
  return rasterLayerState;
};

const convertControlLayerToInpaintMask = (
  newId: string,
  rasterLayerState: CanvasControlLayerState,
  overrides?: PartialDeep<CanvasInpaintMaskState>
): CanvasInpaintMaskState => {
  const { name, objects, position } = rasterLayerState;
  const inpaintMaskState = getInpaintMaskState(newId, {
    name,
    objects,
    position,
  });
  merge(inpaintMaskState, overrides);
  return inpaintMaskState;
};

const convertControlLayerToRegionalGuidance = (
  newId: string,
  rasterLayerState: CanvasControlLayerState,
  overrides?: PartialDeep<CanvasRegionalGuidanceState>
): CanvasRegionalGuidanceState => {
  const { name, objects, position } = rasterLayerState;
  const regionalGuidanceState = getRegionalGuidanceState(newId, {
    name,
    objects,
    position,
  });
  merge(regionalGuidanceState, overrides);
  return regionalGuidanceState;
};

const convertInpaintMaskToRegionalGuidance = (
  newId: string,
  inpaintMaskState: CanvasInpaintMaskState,
  overrides?: PartialDeep<CanvasRegionalGuidanceState>
): CanvasRegionalGuidanceState => {
  const { name, objects, position } = inpaintMaskState;
  const regionalGuidanceState = getRegionalGuidanceState(newId, {
    name,
    objects,
    position,
  });
  merge(regionalGuidanceState, overrides);
  return regionalGuidanceState;
};

const convertRegionalGuidanceToInpaintMask = (
  newId: string,
  regionalGuidanceState: CanvasRegionalGuidanceState,
  overrides?: PartialDeep<CanvasInpaintMaskState>
): CanvasInpaintMaskState => {
  const { name, objects, position } = regionalGuidanceState;
  const inpaintMaskState = getInpaintMaskState(newId, {
    name,
    objects,
    position,
  });
  merge(inpaintMaskState, overrides);
  return inpaintMaskState;
};

/**
 * Supported conversions:
 * - Raster Layer -> Control Layer
 * - Raster Layer -> Inpaint Mask
 * - Raster Layer -> Regional Guidance
 * - Control Layer -> Control Layer
 * - Control Layer -> Inpaint Mask
 * - Control Layer -> Regional Guidance
 * - Inpaint Mask -> Regional Guidance
 * - Regional Guidance -> Inpaint Mask
 */
export const converters = {
  rasterLayer: {
    toControlLayer: convertRasterLayerToControlLayer,
    toInpaintMask: convertRasterLayerToInpaintMask,
    toRegionalGuidance: convertRasterLayerToRegionalGuidance,
  },
  controlLayer: {
    toRasterLayer: convertControlLayerToRasterLayer,
    toInpaintMask: convertControlLayerToInpaintMask,
    toRegionalGuidance: convertControlLayerToRegionalGuidance,
  },
  inpaintMask: {
    toRegionalGuidance: convertInpaintMaskToRegionalGuidance,
  },
  regionalGuidance: {
    toInpaintMask: convertRegionalGuidanceToInpaintMask,
  },
};
