import type { GalleryImage } from '@features/gallery';
import type { GenerateModelConfig, GenerateWidgetValues, VaeModelConfig } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';
import type { WorkbenchCommands } from '@workbench/workbenchStore';

import { galleryImages } from '@features/gallery';
import {
  getDefaultGenerateSettings,
  getEffectiveReferenceImage,
  isSupportedGenerateModel,
  isVaeModelConfig,
  normalizeGenerateWidgetValues,
} from '@features/generation/settings';

import {
  buildImageRecallSettings,
  getImageRecallMessage,
  getImageRecallTitle,
  type ImageRecallKind,
} from './imageRecall';

const imageMetadataRequests = new Map<string, Promise<unknown>>();

const toErrorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

const loadImageMetadata = (imageName: string): Promise<unknown> => {
  const cachedRequest = imageMetadataRequests.get(imageName);

  if (cachedRequest) {
    return cachedRequest;
  }

  const request = galleryImages.metadata(imageName).catch((error: unknown) => {
    imageMetadataRequests.delete(imageName);
    throw error;
  });

  imageMetadataRequests.set(imageName, request);

  return request;
};

export const getCurrentGenerateValues = ({
  generateValues,
  supportedModels,
}: {
  generateValues: Record<string, unknown>;
  supportedModels: GenerateModelConfig[];
}) => {
  const normalizedValues = normalizeGenerateWidgetValues(generateValues);

  if (normalizedValues) {
    return normalizedValues;
  }

  const fallbackModelKey = typeof generateValues.modelKey === 'string' ? generateValues.modelKey : null;
  const fallbackModel = supportedModels.find((model) => model.key === fallbackModelKey) ?? supportedModels[0];

  return fallbackModel
    ? ({
        ...getDefaultGenerateSettings(fallbackModel),
        ...generateValues,
        model: fallbackModel,
        modelKey: fallbackModel.key,
      } as GenerateWidgetValues)
    : null;
};

export const executeImageRecall = async ({
  commands,
  generateValues,
  getGenerateValues,
  image,
  kind,
  models,
  projectId,
}: {
  commands: Pick<WorkbenchCommands, 'generation' | 'notifications'>;
  generateValues: Record<string, unknown>;
  getGenerateValues?: () => Record<string, unknown>;
  image: GalleryImage;
  kind: ImageRecallKind;
  models: ModelConfig[];
  projectId?: string;
}): Promise<boolean> => {
  const supportedModels = models.filter(isSupportedGenerateModel);
  const vaeModels = models.filter(isVaeModelConfig).map((model) => model as VaeModelConfig);
  const currentGenerateValues = getCurrentGenerateValues({
    generateValues: getGenerateValues?.() ?? generateValues,
    supportedModels,
  });

  try {
    if (!currentGenerateValues) {
      commands.notifications.add({
        kind: 'info',
        message: 'Select a supported Generate model first.',
        title: 'Cannot recall image data',
      });
      return false;
    }

    const metadata = kind === 'dimensions' ? null : await loadImageMetadata(image.imageName);
    const result = buildImageRecallSettings({
      currentValues: currentGenerateValues,
      image,
      kind,
      metadata,
      models,
      supportedModels,
      vaeModels,
    });

    if (!result) {
      commands.notifications.add({
        kind: 'info',
        message: 'This image does not include supported Generate metadata.',
        title: 'No recallable image data',
      });
      return false;
    }

    if (result.fields.includes('referenceImages')) {
      const effectiveNames = result.values.referenceImages.flatMap((referenceImage) =>
        referenceImage.config.image ? [getEffectiveReferenceImage(referenceImage.config.image).image_name] : []
      );
      const availableNames = new Set(
        (await galleryImages.resolveMany([...new Set(effectiveNames)])).map(
          (availableImage) => availableImage.imageName
        )
      );
      result.values.referenceImages = result.values.referenceImages.filter((referenceImage) => {
        const referenceAsset = referenceImage.config.image;
        return referenceAsset && availableNames.has(getEffectiveReferenceImage(referenceAsset).image_name);
      });

      if (result.values.referenceImages.length === 0) {
        result.fields = result.fields.filter((field) => field !== 'referenceImages');
      }
    }

    if (result.fields.length === 0) {
      commands.notifications.add({
        kind: 'info',
        message: 'This image does not include supported Generate metadata.',
        title: 'No recallable image data',
      });
      return false;
    }

    commands.generation.setSettings(result.values, projectId);
    commands.notifications.add({
      kind: 'success',
      message: getImageRecallMessage(result.fields),
      title: getImageRecallTitle(kind),
    });
    return true;
  } catch (error: unknown) {
    commands.notifications.reportError({
      area: 'image-recall',
      message: toErrorMessage(error),
      namespace: 'generation',
      projectId,
    });
    return false;
  }
};
