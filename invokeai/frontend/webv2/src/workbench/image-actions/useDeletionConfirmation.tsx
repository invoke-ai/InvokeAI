import type { GalleryItemRef } from '@features/gallery/contracts';
import type { ReactNode } from 'react';

import { Checkbox, Stack, Text } from '@chakra-ui/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { patchWorkbenchPreferences } from '@workbench/settings/store';
import { useCallback, useId, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

export type RequestDeletionConfirmation = (
  itemRefs: readonly GalleryItemRef[],
  executeDeletion: () => Promise<void>,
  /** Where focus goes when the dialog closes; without it, back to the control that opened it. */
  returnFocus?: () => HTMLElement | null
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
  // Kept past the request: the dialog closes after the pending deletion clears, and resolves focus as it does.
  const [returnFocus, setReturnFocus] = useState<(() => HTMLElement | null) | null>(null);
  const pendingDeletionRef = useRef<PendingDeletion | null>(null);
  const deletionInFlightRef = useRef(false);
  // Read only when deletion is confirmed, so ticking it and then cancelling changes nothing.
  const [skipFutureConfirmations, setSkipFutureConfirmations] = useState(false);
  const skipHintId = useId();

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

  const requestDeletionConfirmation = useCallback<RequestDeletionConfirmation>(
    (itemRefs, executeDeletion, getReturnFocus) => {
      if (pendingDeletionRef.current) {
        return Promise.resolve();
      }

      return new Promise<void>((resolve) => {
        const pending = { executeDeletion, itemRefs: [...itemRefs], resolve };

        pendingDeletionRef.current = pending;
        setSkipFutureConfirmations(false);
        setPendingDeletion(pending);
        setReturnFocus(() => getReturnFocus ?? null);
      });
    },
    []
  );

  const handleConfirm = useCallback(async () => {
    const pending = pendingDeletionRef.current;

    if (!pending || deletionInFlightRef.current) {
      return;
    }

    deletionInFlightRef.current = true;

    if (skipFutureConfirmations) {
      void patchWorkbenchPreferences({ confirmImageDeletion: false });
    }

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
  }, [skipFutureConfirmations]);

  const handleSkipChange = useCallback((details: { checked: boolean | 'indeterminate' }) => {
    setSkipFutureConfirmations(details.checked === true);
  }, []);

  useMountEffect(() => settlePendingDeletion);

  const itemCount = pendingDeletion?.itemRefs.length ?? 0;
  const isImageOnly = pendingDeletion?.itemRefs.every((item) => item.kind === 'image') ?? false;
  const titleKey = isImageOnly ? 'widgets.gallery.deleteImagesConfirmTitle' : 'widgets.gallery.deleteItemsConfirmTitle';
  const bodyKey = isImageOnly ? 'widgets.gallery.deleteImagesConfirmBody' : 'widgets.gallery.deleteItemsConfirmBody';

  return {
    dialog: (
      <ConfirmDialog
        body={
          <>
            <Text fontSize="md">{t(bodyKey, { count: itemCount })}</Text>
            <Stack gap="1" pt="1">
              <Checkbox.Root checked={skipFutureConfirmations} size="sm" onCheckedChange={handleSkipChange}>
                <Checkbox.HiddenInput aria-describedby={skipHintId} />
                <Checkbox.Control>
                  <Checkbox.Indicator />
                </Checkbox.Control>
                <Checkbox.Label>{t('widgets.gallery.deleteConfirmSkip')}</Checkbox.Label>
              </Checkbox.Root>
              <Text color="fg.muted" fontSize="sm" id={skipHintId} ps="4.5">
                {t('widgets.gallery.deleteConfirmSkipHint')}
              </Text>
            </Stack>
          </>
        }
        confirmLabel={t('widgets.gallery.deleteConfirmLabel')}
        finalFocusEl={returnFocus ?? undefined}
        isOpen={pendingDeletion !== null}
        title={t(titleKey, { count: itemCount })}
        onClose={settlePendingDeletion}
        onConfirm={handleConfirm}
      />
    ),
    requestDeletionConfirmation,
  };
};
