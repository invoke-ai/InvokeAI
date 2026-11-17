import type {
  QueueBackendItem,
  QueueCounts,
  QueueItemDetailReadModel,
  QueueItemIdsReadModel,
  QueueItemReadModel,
  QueueNodeFieldValue,
  QueueStatusReadModel,
} from '@features/queue/core/types';

import { getOutputImageNames } from '@platform/core/outputImages';
import { normalizeServerTimestamp } from '@platform/time/serverTimestamp';

import type { QueueItemStatusChangedEvent } from './events';
import type { QueueItemStatusChange } from './queries';
import type {
  QueueAndProcessorStatusDTO,
  QueueItemIdsResultDTO,
  QueueNodeFieldValueDTO,
  QueueServerItemDTO,
  QueueStatusCountsDTO,
} from './serverTypes';

const mapNodeFieldValue = (dto: QueueNodeFieldValueDTO): QueueNodeFieldValue => ({
  fieldName: dto.field_name,
  nodePath: dto.node_path,
  value: dto.value && typeof dto.value === 'object' ? { imageName: dto.value.image_name } : dto.value,
});

/** The server sends naive UTC; absent and null stay as they are. */
const normalizeOptionalTimestamp = <T extends string | null | undefined>(value: T): T =>
  (typeof value === 'string' ? normalizeServerTimestamp(value) : value) as T;

const getResultImageNames = (dto: QueueServerItemDTO): string[] => {
  return Object.values(dto.session?.results ?? {}).flatMap(getOutputImageNames);
};

export const mapQueueItemDTO = (dto: QueueServerItemDTO): QueueItemReadModel => ({
  batchId: dto.batch_id,
  completedAt: normalizeOptionalTimestamp(dto.completed_at),
  createdAt: normalizeServerTimestamp(dto.created_at),
  destination: dto.destination,
  device: dto.device,
  errorMessage: dto.error_message,
  errorTraceback: dto.error_traceback,
  errorType: dto.error_type,
  fieldValues:
    dto.field_values === null || dto.field_values === undefined
      ? dto.field_values
      : dto.field_values.map(mapNodeFieldValue),
  id: dto.item_id,
  origin: dto.origin,
  parentItemId: dto.parent_item_id,
  priority: dto.priority,
  resultImageNames: getResultImageNames(dto),
  retriedFromItemId: dto.retried_from_item_id,
  sessionId: dto.session_id,
  startedAt: normalizeOptionalTimestamp(dto.started_at),
  status: dto.status,
  statusSequence: dto.status_sequence,
  updatedAt: normalizeServerTimestamp(dto.updated_at),
  userDisplayName: dto.user_display_name,
  userEmail: dto.user_email,
  userId: dto.user_id,
  workflowName: typeof dto.workflow?.name === 'string' ? dto.workflow.name : undefined,
});

/** The saved execution data alone; the live row keeps its own narrower mapping. */
export const mapQueueItemDetailDTO = (dto: QueueServerItemDTO): QueueItemDetailReadModel => ({
  errorMessage: dto.error_message ?? null,
  errorTraceback: dto.error_traceback ?? null,
  errorType: dto.error_type ?? null,
  graph: dto.session?.graph ?? {},
  id: dto.item_id,
  priority: dto.priority ?? null,
  results: dto.session?.results ?? {},
  sessionId: dto.session_id,
  status: dto.status,
  workflow: dto.workflow ?? null,
});

export const mapQueueBackendItemDTO = (dto: QueueServerItemDTO): QueueBackendItem => ({
  batchId: dto.batch_id,
  destination: dto.destination,
  errorMessage: dto.error_message,
  errorType: dto.error_type,
  id: dto.item_id,
  origin: dto.origin,
  status: dto.status,
});

export const mapQueueItemIdsDTO = (dto: QueueItemIdsResultDTO): QueueItemIdsReadModel => ({
  itemIds: dto.item_ids,
});

const mapQueueCountsDTO = (dto: QueueStatusCountsDTO): QueueCounts => ({
  batchId: dto.batch_id,
  canceled: dto.canceled,
  completed: dto.completed,
  failed: dto.failed,
  inProgress: dto.in_progress,
  itemId: dto.item_id,
  pending: dto.pending,
  queueId: dto.queue_id,
  sessionId: dto.session_id,
  total: dto.total,
  userInProgress: dto.user_in_progress,
  userPending: dto.user_pending,
  waiting: dto.waiting,
});

export const mapQueueStatusDTO = (dto: QueueAndProcessorStatusDTO): QueueStatusReadModel => ({
  processor: {
    isProcessing: dto.processor.is_processing,
    isStarted: dto.processor.is_started,
  },
  queue: mapQueueCountsDTO(dto.queue),
});

/** The item fields a status event carries, and the whole queue's counts after the change. */
export const mapQueueItemStatusChangedEvent = (event: QueueItemStatusChangedEvent): QueueItemStatusChange => ({
  item: {
    batchId: event.batch_id,
    completedAt: normalizeOptionalTimestamp(event.completed_at),
    createdAt: normalizeServerTimestamp(event.created_at),
    destination: event.destination,
    errorMessage: event.error_message,
    errorTraceback: event.error_traceback,
    errorType: event.error_type,
    id: event.item_id,
    origin: event.origin,
    resultImageNames: [],
    sessionId: event.session_id,
    startedAt: normalizeOptionalTimestamp(event.started_at),
    status: event.status,
    statusSequence: event.status_sequence,
    updatedAt: normalizeServerTimestamp(event.updated_at),
    userId: event.user_id,
  },
  queueCounts: mapQueueCountsDTO(event.queue_status),
});
