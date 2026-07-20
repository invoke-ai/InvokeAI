import type { ReactNode } from 'react';

import { createExternalStore } from '@platform/state/externalStore';

export interface QueueConfirmation {
  body: ReactNode;
  confirmLabel: string;
  onConfirm: () => Promise<void> | void;
  title: string;
}

const queueConfirmationStore = createExternalStore<{ confirmation: QueueConfirmation | null }>({ confirmation: null });

export const requestQueueConfirmation = (confirmation: QueueConfirmation): void => {
  queueConfirmationStore.setSnapshot({ confirmation });
};

export const clearQueueConfirmation = (): void => {
  queueConfirmationStore.setSnapshot({ confirmation: null });
};

export const getQueueConfirmation = (): QueueConfirmation | null => queueConfirmationStore.getSnapshot().confirmation;

export const useQueueConfirmation = (): QueueConfirmation | null =>
  queueConfirmationStore.useSelector((snapshot) => snapshot.confirmation, Object.is);
