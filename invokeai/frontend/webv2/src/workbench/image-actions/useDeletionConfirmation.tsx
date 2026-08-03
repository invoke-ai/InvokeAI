import type { GalleryItemRef } from '@features/gallery/contracts';
import type { ReactNode } from 'react';

import { useMountEffect } from '@platform/react/useMountEffect';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { useCallback, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

export type RequestDeletionConfirmation = (
  itemRefs: readonly GalleryItemRef[],
  executeDeletion: () => Promise<void>
) => Promise<void>;

interface PendingDeletion {
  executeDeletion: () => Promise<void>;
  itemRefs: readonly GalleryItemRef[];
  resolve: () => void;
}

export const useDeletionConfirmation = (): {
  dialog: ReactNode;
  requestDeletionConfirmation: RequestDeletionConfirmation;
} => {
  const { t } = useTranslation();
  const [pendingDeletion, setPendingDeletion] = useState<PendingDeletion | null>(null);
  const pendingDeletionRef = useRef<PendingDeletion | null>(null);
  const deletionInFlightRef = useRef(false);

  const settlePendingDeletion = useCallback(() => {
    const pending = pendingDeletionRef.current;

    if (!pending) {
      return;
    }

    pendingDeletionRef.current = null;
    deletionInFlightRef.current = false;
    setPendingDeletion(null);
    pending.resolve();
  }, []);

  const requestDeletionConfirmation = useCallback<RequestDeletionConfirmation>((itemRefs, executeDeletion) => {
    if (pendingDeletionRef.current) {
      return Promise.resolve();
    }

    return new Promise<void>((resolve) => {
      const pending = { executeDeletion, itemRefs: [...itemRefs], resolve };

      pendingDeletionRef.current = pending;
      setPendingDeletion(pending);
    });
  }, []);

  const handleConfirm = useCallback(async () => {
    const pending = pendingDeletionRef.current;

    if (!pending || deletionInFlightRef.current) {
      return;
    }

    deletionInFlightRef.current = true;

    try {
      await pending.executeDeletion();
    } finally {
      if (pendingDeletionRef.current === pending) {
        pendingDeletionRef.current = null;
        deletionInFlightRef.current = false;
        setPendingDeletion(null);
        pending.resolve();
      }
    }
  }, []);

  useMountEffect(() => settlePendingDeletion);

  const itemCount = pendingDeletion?.itemRefs.length ?? 0;
  const isImageOnly = pendingDeletion?.itemRefs.every((item) => item.kind === 'image') ?? false;
  const titleKey = isImageOnly ? 'widgets.gallery.deleteImagesConfirmTitle' : 'widgets.gallery.deleteItemsConfirmTitle';
  const bodyKey = isImageOnly ? 'widgets.gallery.deleteImagesConfirmBody' : 'widgets.gallery.deleteItemsConfirmBody';

  return {
    dialog: (
      <ConfirmDialog
        body={t(bodyKey, { count: itemCount })}
        confirmLabel={t('widgets.gallery.deleteConfirmLabel')}
        isOpen={pendingDeletion !== null}
        title={t(titleKey, { count: itemCount })}
        onClose={settlePendingDeletion}
        onConfirm={handleConfirm}
      />
    ),
    requestDeletionConfirmation,
  };
};
