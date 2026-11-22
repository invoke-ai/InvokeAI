import type { GeneratedImageContract } from '@features/gallery';
import type { GenerateWidgetValues } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';

import {
  addReferenceImages,
  createReferenceImageId,
  generatedImageToReferenceImage,
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
  const currentValues = getCurrentGenerateValues({ generateValues, models, supportedModels });

  if (!currentValues) {
    return { status: 'unsupported' };
  }

  const { referenceImages, refusal } = addReferenceImages({
    additions: [{ id: createReferenceImageId(), image: generatedImageToReferenceImage(image) }],
    components: currentValues,
    mode: 'strict',
    model: currentValues.model,
    models,
    referenceImages: currentValues.referenceImages,
  });

  if (refusal) {
    return { status: refusal === 'full' ? 'full' : 'unsupported' };
  }

  return { referenceImages, status: 'appended' };
};
