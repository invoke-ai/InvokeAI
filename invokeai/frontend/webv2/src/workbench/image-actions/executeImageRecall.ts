import type { GalleryImage } from '@workbench/gallery/api';
import type { GenerateModelConfig, GenerateWidgetValues, VaeModelConfig } from '@workbench/generation/types';
import type { ModelConfig } from '@workbench/models/types';
import type { WorkbenchAction } from '@workbench/workbenchState';
import type { Dispatch } from 'react';

import { getGalleryImageMetadata } from '@workbench/gallery/api';
import { getDefaultGenerateSettings, isSupportedGenerateModel } from '@workbench/generation/baseGenerationPolicies';
import { isVaeModelConfig, normalizeGenerateWidgetValues } from '@workbench/generation/settings';

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

  const request = getGalleryImageMetadata(imageName).catch((error: unknown) => {
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
  dispatch,
  generateValues,
  getGenerateValues,
  image,
  kind,
  models,
  projectId,
}: {
  dispatch: Dispatch<WorkbenchAction>;
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
      dispatch({
        kind: 'info',
        message: 'Select a supported Generate model first.',
        title: 'Cannot recall image data',
        type: 'recordNotice',
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
      dispatch({
        kind: 'info',
        message: 'This image does not include supported Generate metadata.',
        title: 'No recallable image data',
        type: 'recordNotice',
      });
      return false;
    }

    dispatch({ projectId, type: 'setGenerateSettings', values: result.values });
    dispatch({
      kind: 'success',
      message: getImageRecallMessage(result.fields),
      title: getImageRecallTitle(kind),
      type: 'recordNotice',
    });
    return true;
  } catch (error: unknown) {
    dispatch({
      area: 'image-recall',
      message: toErrorMessage(error),
      namespace: 'generation',
      projectId,
      type: 'recordError',
    });
    return false;
  }
};
