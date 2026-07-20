import type { GenerateWidgetValues } from '@features/generation/contracts';
import type { QueueItem, QueueState, QueueSubmissionSnapshot } from '@features/queue/contracts';

import type { CanvasStateContractV2 } from './canvas-engine/api';
import type { WidgetInstanceContract, WidgetInstanceId, WidgetStateMap } from './widgetContracts';

/** Workbench-owned persistence context that Queue stores but never interprets. */
export interface WorkbenchQueueSubmissionContext {
  canvas: CanvasStateContractV2;
  generate?: {
    negativePromptNodeId: string;
    positivePromptNodeId: string;
    seedNodeId: string;
    values: GenerateWidgetValues;
  };
  widgetInstances: Record<WidgetInstanceId, WidgetInstanceContract>;
  widgetStates: WidgetStateMap;
}

export type WorkbenchQueueItem = Omit<QueueItem, 'snapshot'> & {
  snapshot: QueueSubmissionSnapshot & WorkbenchQueueSubmissionContext;
};

export type WorkbenchQueueState = Omit<QueueState, 'items'> & { items: WorkbenchQueueItem[] };
