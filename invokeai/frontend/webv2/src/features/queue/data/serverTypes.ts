import type { QueueItemStatus } from '@features/queue/core/types';

/** Private HTTP wire contracts. These names and fields mirror backend JSON. */
export interface QueueNodeFieldValueDTO {
  field_name: string;
  node_path: string;
  value: string | number | { image_name?: string } | null;
}

export interface QueueServerItemDTO {
  batch_id: string;
  completed_at?: string | null;
  created_at: string;
  destination?: string | null;
  error_message?: string | null;
  error_traceback?: string | null;
  error_type?: string | null;
  field_values?: QueueNodeFieldValueDTO[] | null;
  item_id: number;
  origin?: string | null;
  retried_from_item_id?: number | null;
  session?: {
    prepared_source_mapping?: Record<string, string>;
    results?: Record<string, unknown>;
  };
  session_id: string;
  started_at?: string | null;
  status: QueueItemStatus;
  updated_at: string;
}

export interface QueueImageDTO {
  height: number;
  image_name: string;
  image_url: string;
  is_intermediate: boolean;
  thumbnail_url: string;
  width: number;
}

export interface QueueStatusCountsDTO {
  batch_id?: string | null;
  canceled: number;
  completed: number;
  failed: number;
  in_progress: number;
  item_id?: number | null;
  pending: number;
  queue_id: string;
  session_id?: string | null;
  total: number;
}

export interface QueueProcessorStatusDTO {
  is_processing: boolean;
  is_started: boolean;
}

export interface QueueAndProcessorStatusDTO {
  processor: QueueProcessorStatusDTO;
  queue: QueueStatusCountsDTO;
}

export interface QueueItemIdsResultDTO {
  item_ids: number[];
  total_count: number;
}
