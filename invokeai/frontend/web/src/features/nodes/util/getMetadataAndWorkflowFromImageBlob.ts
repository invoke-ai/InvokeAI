import * as png from '@stevebel/png';
import { logger } from 'app/logging/logger';
import {
  ImageMetadataAndWorkflow,
  zCoreMetadata,
  zWorkflow,
} from 'features/nodes/types/types';
import { get } from 'lodash-es';

export const getMetadataAndWorkflowFromImageBlob = async (
  image: Blob
): Promise<ImageMetadataAndWorkflow> => {
  const data: ImageMetadataAndWorkflow = {};
  try {
    const buffer = await image.arrayBuffer();
    const text = png.decode(buffer).text;
    const rawMetadata = get(text, 'invokeai_metadata');
    const rawWorkflow = get(text, 'invokeai_workflow');
    if (rawMetadata) {
      try {
        data.metadata = zCoreMetadata.parse(JSON.parse(rawMetadata));
      } catch {
        // no-op
      }
    }
    if (rawWorkflow) {
      try {
        data.workflow = zWorkflow.parse(JSON.parse(rawWorkflow));
      } catch {
        // no-op
      }
    }
  } catch {
    logger('nodes').warn('Unable to parse image');
  }
  return data;
};
