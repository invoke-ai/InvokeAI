import type { QueueItemStatus } from '@features/queue/core/types';

/**
 * Presentation metadata for each backend queue status: the human label, the
 * Chakra `colorPalette` for badges, and the dot color used in list rows and the
 * NOW & NEXT card. One table so status styling stays consistent everywhere.
 */
export interface QueueStatusMeta {
  labelKey: string;
  colorPalette: string;
  dotColor: string;
}

const STATUS_META: Record<QueueItemStatus, QueueStatusMeta> = {
  canceled: { colorPalette: 'orange', dotColor: 'orange.solid', labelKey: 'common.status.canceled' },
  completed: { colorPalette: 'green', dotColor: 'green.solid', labelKey: 'common.status.completed' },
  failed: { colorPalette: 'red', dotColor: 'red.solid', labelKey: 'common.status.failed' },
  in_progress: { colorPalette: 'accent', dotColor: 'accent.solid', labelKey: 'common.generating' },
  pending: { colorPalette: 'gray', dotColor: 'fg.muted', labelKey: 'common.status.pending' },
};

export const getStatusMeta = (status: QueueItemStatus): QueueStatusMeta => STATUS_META[status];
