import type { GeneratedImageContract } from '@features/gallery';
import type { GenerateWidgetValues } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';

import {
  createReferenceImageId,
  generatedImageToReferenceImage,
  getDefaultReferenceImageConfig,
  getMaxReferenceImages,
  isReferenceImageSupported,
  isSupportedGenerateModel,
} from '@features/generation/settings';

import { getCurrentGenerateValues } from './executeImageRecall';

export type AppendReferenceImageResult =
  | { status: 'appended'; referenceImages: GenerateWidgetValues['referenceImages'] }
  | { status: 'full' | 'unsupported' };

/**
 * Builds the generate widget's next `referenceImages` array with `image`
 * appended as a new reference image for the current model. Pure: the caller
 * patches the widget values on 'appended'.
 */
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

  const referenceImage = {
    config: getDefaultReferenceImageConfig(currentValues.model, models, generatedImageToReferenceImage(image)),
    id: createReferenceImageId(),
    isEnabled: true,
  };

  return { referenceImages: [...currentValues.referenceImages, referenceImage], status: 'appended' };
};
