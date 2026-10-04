import type {
  QueueBackendItem,
  QueueItemIdsReadModel,
  QueueItemReadModel,
  QueueNodeFieldValue,
  QueueStatusReadModel,
} from '@features/queue/core/types';

import { getOutputImageNames } from '@platform/core/outputImages';

import type {
  QueueAndProcessorStatusDTO,
  QueueItemIdsResultDTO,
  QueueNodeFieldValueDTO,
  QueueServerItemDTO,
} from './serverTypes';

const mapNodeFieldValue = (dto: QueueNodeFieldValueDTO): QueueNodeFieldValue => ({
  fieldName: dto.field_name,
  nodePath: dto.node_path,
  value: dto.value && typeof dto.value === 'object' ? { imageName: dto.value.image_name } : dto.value,
});

const getResultImageNames = (dto: QueueServerItemDTO): string[] => {
  return Object.values(dto.session?.results ?? {}).flatMap(getOutputImageNames);
};

export const mapQueueItemDTO = (dto: QueueServerItemDTO): QueueItemReadModel => ({
  batchId: dto.batch_id,
  completedAt: dto.completed_at,
  createdAt: dto.created_at,
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
  resultImageNames: getResultImageNames(dto),
  retriedFromItemId: dto.retried_from_item_id,
  sessionId: dto.session_id,
  startedAt: dto.started_at,
  status: dto.status,
  updatedAt: dto.updated_at,
  userDisplayName: dto.user_display_name,
  userEmail: dto.user_email,
  userId: dto.user_id,
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
  totalCount: dto.total_count,
});

export const mapQueueStatusDTO = (dto: QueueAndProcessorStatusDTO): QueueStatusReadModel => ({
  processor: {
    isProcessing: dto.processor.is_processing,
    isStarted: dto.processor.is_started,
  },
  queue: {
    batchId: dto.queue.batch_id,
    canceled: dto.queue.canceled,
    completed: dto.queue.completed,
    failed: dto.queue.failed,
    inProgress: dto.queue.in_progress,
    itemId: dto.queue.item_id,
    pending: dto.queue.pending,
    queueId: dto.queue.queue_id,
    sessionId: dto.queue.session_id,
    total: dto.queue.total,
    userInProgress: dto.queue.user_in_progress,
    userPending: dto.queue.user_pending,
    waiting: dto.queue.waiting,
  },
});
