/**
 * Exclude submitted input media from results: primitive nodes echo inputs, while generated outputs receive fresh
 * server names. Inputs live in compiled graph nodes and, for workflow Image Batches, in the submission's batch data,
 * which the backend substitutes into nodes per session.
 */

import { isQueueWorkflowBatchDatum } from './promptBatch';

export interface SubmissionInputMediaNames {
  imageNames: ReadonlySet<string>;
  videoNames: ReadonlySet<string>;
}

/**
 * Collect image_name/video_name strings from a compiled submission's graph nodes and the image items of its well-formed
 * batch datums. Malformed or legacy persisted submissions, and malformed batch datums, yield no names: restored
 * snapshots are not validated, and a stray name must not filter out a real result.
 */
export const collectSubmissionInputMediaNames = (submission: unknown): SubmissionInputMediaNames => {
  const imageNames = new Set<string>();
  const videoNames = new Set<string>();

  const visit = (value: unknown): void => {
    if (Array.isArray(value)) {
      for (const entry of value) {
        visit(entry);
      }
      return;
    }
    if (!value || typeof value !== 'object') {
      return;
    }

    const imageName = (value as { image_name?: unknown }).image_name;
    if (typeof imageName === 'string') {
      imageNames.add(imageName);
    }
    const videoName = (value as { video_name?: unknown }).video_name;
    if (typeof videoName === 'string') {
      videoNames.add(videoName);
    }

    for (const entry of Object.values(value)) {
      visit(entry);
    }
  };

  if (!submission || typeof submission !== 'object') {
    return { imageNames, videoNames };
  }

  const { batchData, graph } = submission as { batchData?: unknown; graph?: unknown };
  const nodes = graph && typeof graph === 'object' ? (graph as { nodes?: unknown }).nodes : undefined;
  if (nodes && typeof nodes === 'object') {
    for (const node of Object.values(nodes)) {
      visit(node);
    }
  }
  if (Array.isArray(batchData)) {
    for (const group of batchData) {
      if (!Array.isArray(group)) {
        continue;
      }
      for (const datum of group) {
        if (!isQueueWorkflowBatchDatum(datum)) {
          continue;
        }
        for (const item of datum.items) {
          if (typeof item === 'object') {
            imageNames.add(item.image_name);
          }
        }
      }
    }
  }

  return { imageNames, videoNames };
};
