import { Portal, Stack, Text } from '@chakra-ui/react';
import { useExitRetainedValue } from '@platform/react/useExitRetainedValue';
import { useCallback, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

import { Button, CloseButton } from './Button';
import { Dialog } from './Dialog';

/** Closes after confirmation even on error; callers must report failures. */
export const ConfirmDialog = ({
  body,
  cancelLabel,
  confirmLabel,
  finalFocusEl,
  initialFocusEl,
  isDestructive = true,
  isConfirmDisabled = false,
  isOpen,
  onClose,
  onConfirm,
  onExitComplete,
  title,
}: {
  body: ReactNode;
  /**
   * Names the way out; defaults to "Cancel". Override it whenever the confirmed action is itself a cancel ("Cancel
   * queue item"), so the two buttons never read alike ("Keep running").
   */
  cancelLabel?: string;
  confirmLabel: string;
  /** Where focus returns on close when the opener may no longer exist, e.g. a row the confirmed action deleted. */
  finalFocusEl?: () => HTMLElement | null;
  /**
   * Where focus starts instead of the dismiss button, such as a typed-confirmation field: safe only when the confirm
   * stays disabled until the body is filled in.
   */
  initialFocusEl?: () => HTMLElement | null;
  isDestructive?: boolean;
  /** Keep confirmation unavailable until the caller has prepared a valid confirmation target. */
  isConfirmDisabled?: boolean;
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => Promise<void> | void;
  /** After the close animation; hosts that retain the dialog's subject release it here. */
  onExitComplete?: () => void;
  title: string;
}) => {
  const { t } = useTranslation();
  const [isPending, setIsPending] = useState(false);
  const isPendingRef = useRef(false);
  const cancelRef = useRef<HTMLButtonElement>(null);
  const resolvedCancelLabel = cancelLabel ?? t('common.cancel');
  // Hosts often clear the subject the dialog describes as it closes; keep the text it showed while it animates out.
  const live = useMemo(
    () => ({ body, cancelLabel: resolvedCancelLabel, confirmLabel, isDestructive, title }),
    [body, confirmLabel, isDestructive, resolvedCancelLabel, title]
  );
  const shown = useExitRetainedValue(isOpen ? live : null);
  const text = shown.value ?? live;
  const { release } = shown;
  const handleExitComplete = useCallback(() => {
    release();
    onExitComplete?.();
  }, [onExitComplete, release]);

  const handleConfirm = useCallback(async () => {
    if (isPendingRef.current || isConfirmDisabled) {
      return;
    }

    isPendingRef.current = true;
    setIsPending(true);

    try {
      await onConfirm();
    } finally {
      isPendingRef.current = false;
      setIsPending(false);
      onClose();
    }
  }, [isConfirmDisabled, onClose, onConfirm]);

  const handleClose = useCallback(() => {
    if (!isPendingRef.current) {
      onClose();
    }
  }, [onClose]);

  const handleOpenChange = useCallback(
    (event: { open: boolean }) => {
      if (!event.open) {
        handleClose();
      }
    },
    [handleClose]
  );

  const handleConfirmClick = useCallback(() => {
    void handleConfirm();
  }, [handleConfirm]);
  // Enter in a text field of the body confirms, as a form would, but only while confirming is allowed.
  const handleBodyKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      if (
        event.key === 'Enter' &&
        !event.nativeEvent.isComposing &&
        event.target instanceof HTMLInputElement &&
        (event.target.type === 'text' || event.target.type === 'search')
      ) {
        event.preventDefault();
        void handleConfirm();
      }
    },
    [handleConfirm]
  );

  // A destructive confirmation opens on its dismiss button rather than the dialog's default target (the close button
  // for an alertdialog), so a reflexive Enter or Space cancels and never commits or flips an option in the body.
  const getInitialFocus = useCallback(() => cancelRef.current, []);

  return (
    <Dialog.Root
      closeOnEscape={!isPending}
      closeOnInteractOutside={!isPending}
      finalFocusEl={finalFocusEl}
      initialFocusEl={initialFocusEl ?? (text.isDestructive ? getInitialFocus : undefined)}
      open={isOpen}
      role="alertdialog"
      size="sm"
      onExitComplete={handleExitComplete}
      onOpenChange={handleOpenChange}
    >
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content>
            <Dialog.Header>
              <Dialog.Title>{text.title}</Dialog.Title>
            </Dialog.Header>
            <Dialog.Body onKeyDown={handleBodyKeyDown}>
              <Stack gap="2">
                {typeof text.body === 'string' ? <Text fontSize="md">{text.body}</Text> : text.body}
              </Stack>
            </Dialog.Body>
            <Dialog.Footer>
              <Button ref={cancelRef} disabled={isPending} variant="ghost" onClick={handleClose}>
                {text.cancelLabel}
              </Button>
              <Button
                colorPalette={text.isDestructive ? 'red' : 'accent'}
                disabled={isPending || isConfirmDisabled}
                loading={isPending}
                variant="solid"
                onClick={handleConfirmClick}
              >
                {text.confirmLabel}
              </Button>
            </Dialog.Footer>
            <Dialog.CloseTrigger asChild>
              <CloseButton disabled={isPending} />
            </Dialog.CloseTrigger>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};
