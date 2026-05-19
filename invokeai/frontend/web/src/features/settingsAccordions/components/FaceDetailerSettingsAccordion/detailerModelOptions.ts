import type { ComboboxOption } from '@invoke-ai/ui-library';

export const DETAILER_DINO_MODELS = ['grounding-dino-tiny', 'grounding-dino-base'] as const;

export const DETAILER_SAM_MODELS = [
  'segment-anything-2-small',
  'segment-anything-2-tiny',
  'segment-anything-2-base',
  'segment-anything-2-large',
  'segment-anything-base',
  'segment-anything-large',
  'segment-anything-huge',
] as const;

type TFunction = (key: string) => string;

export const getDetailerDinoModelLabelKey = (model: (typeof DETAILER_DINO_MODELS)[number]) =>
  `parameters.faceDetailer.dinoModels.${model}`;

export const getDetailerSamModelLabelKey = (model: (typeof DETAILER_SAM_MODELS)[number]) =>
  `parameters.faceDetailer.samModels.${model}`;

export const getDetailerDinoModelOptions = (t: TFunction): ComboboxOption[] =>
  DETAILER_DINO_MODELS.map((model) => ({ label: t(getDetailerDinoModelLabelKey(model)), value: model }));

export const getDetailerSamModelOptions = (t: TFunction): ComboboxOption[] =>
  DETAILER_SAM_MODELS.map((model) => ({ label: t(getDetailerSamModelLabelKey(model)), value: model }));
