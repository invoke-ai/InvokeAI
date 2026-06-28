import type { BackendQueueItemStatus } from '@workbench/backend/events';

/**
 * Presentation metadata for each backend queue status: the human label, the
 * Chakra `colorPalette` for badges, and the dot color used in list rows and the
 * NOW & NEXT card. One table so status styling stays consistent everywhere.
 */
export interface QueueStatusMeta {
  label: string;
  colorPalette: string;
  dotColor: string;
}

const STATUS_META: Record<BackendQueueItemStatus, QueueStatusMeta> = {
  canceled: { colorPalette: 'orange', dotColor: 'orange.solid', label: 'Canceled' },
  completed: { colorPalette: 'green', dotColor: 'green.solid', label: 'Completed' },
  failed: { colorPalette: 'red', dotColor: 'red.solid', label: 'Failed' },
  in_progress: { colorPalette: 'accent', dotColor: 'accent.solid', label: 'Generating' },
  pending: { colorPalette: 'gray', dotColor: 'fg.muted', label: 'Pending' },
};

export const getStatusMeta = (status: BackendQueueItemStatus): QueueStatusMeta => STATUS_META[status];
