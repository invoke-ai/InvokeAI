import type { GenerateWidgetValues } from '@features/generation/contracts';
import type { QueueItem, QueueState, QueueSubmissionSnapshot } from '@features/queue/contracts';
import type { VideoWidgetValues } from '@features/video';

import type { CanvasStateContractV3 } from './canvas-engine/api';

export interface QueueCanvasSessionRef {
  document: Pick<CanvasStateContractV3['document'], 'bbox' | 'height' | 'width'>;
  documentRevision: number;
}

export interface WorkbenchQueueRecallValues {
  generateValues?: GenerateWidgetValues;
  videoValues?: VideoWidgetValues;
}

export interface WorkbenchQueueSubmissionContext {
  canvas: QueueCanvasSessionRef;
  recall?: WorkbenchQueueRecallValues;
}

export type WorkbenchQueueItem = Omit<QueueItem, 'snapshot'> & {
  snapshot: QueueSubmissionSnapshot & WorkbenchQueueSubmissionContext;
};

export type WorkbenchQueueState = Omit<QueueState, 'items'> & { items: WorkbenchQueueItem[] };
