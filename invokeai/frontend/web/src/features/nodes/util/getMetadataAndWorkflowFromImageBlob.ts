import * as png from '@stevebel/png';
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
  const buffer = await image.arrayBuffer();
  const text = png.decode(buffer).text;

  const rawMetadata = get(text, 'invokeai_metadata');
  if (rawMetadata) {
    const metadataResult = zCoreMetadata.safeParse(JSON.parse(rawMetadata));
    if (metadataResult.success) {
      data.metadata = metadataResult.data;
    }
  }

  const rawWorkflow = get(text, 'invokeai_workflow');
  if (rawWorkflow) {
    const workflowResult = zWorkflow.safeParse(JSON.parse(rawWorkflow));
    if (workflowResult.success) {
      data.workflow = workflowResult.data;
    }
  }

  return data;
};
