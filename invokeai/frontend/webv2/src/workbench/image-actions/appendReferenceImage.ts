import type { GeneratedImageContract } from '@features/gallery';
import type { GenerateWidgetValues } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';

import {
  createReferenceImageId,
  generatedImageToReferenceImage,
  getMaxReferenceImages,
  isReferenceImageSupported,
  isSupportedGenerateModel,
  withAddedReferenceImages,
} from '@features/generation/settings';

import { getCurrentGenerateValues } from './executeImageRecall';

export type AppendReferenceImageResult =
  | { status: 'appended'; referenceImages: GenerateWidgetValues['referenceImages'] }
  | { status: 'full' | 'unsupported' };

export const appendReferenceImage = ({
  generateValues,
  image,
  models,
}: {
  generateValues: Record<string, unknown>;
  image: Pick<GeneratedImageContract, 'height' | 'imageName' | 'width'>;
  models: ModelConfig[];
}): AppendReferenceImageResult => {
  const supportedModels = models.filter(isSupportedGenerateModel);
  const currentValues = getCurrentGenerateValues({ generateValues, supportedModels });

  if (!currentValues || !isReferenceImageSupported(currentValues.model)) {
    return { status: 'unsupported' };
  }
  if (currentValues.referenceImages.length >= getMaxReferenceImages(currentValues.model)) {
    return { status: 'full' };
  }

  return {
    referenceImages: withAddedReferenceImages(currentValues.model, models, currentValues.referenceImages, [
      { id: createReferenceImageId(), image: generatedImageToReferenceImage(image) },
    ]),
    status: 'appended',
  };
};
