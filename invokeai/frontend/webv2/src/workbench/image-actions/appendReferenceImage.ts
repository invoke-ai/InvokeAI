import type { GeneratedImageContract } from '@features/gallery';
import type { GenerateWidgetValues } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';

import {
  createReferenceImageId,
  generatedImageToReferenceImage,
  getDefaultReferenceImageConfigs,
  getMaxReferenceImages,
  isReferenceImageSupported,
  isSupportedGenerateModel,
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

  const [config] = getDefaultReferenceImageConfigs(
    currentValues.model,
    models,
    [{ image: generatedImageToReferenceImage(image), isEnabled: true }],
    currentValues.referenceImages
  );
  const referenceImage = { config: config!, id: createReferenceImageId(), isEnabled: true };

  return { referenceImages: [...currentValues.referenceImages, referenceImage], status: 'appended' };
};
