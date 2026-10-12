import { Portal, Stack, Text } from '@chakra-ui/react';
import { useExitRetainedValue } from '@platform/react/useExitRetainedValue';
import { useCallback, useMemo, useRef, useState, type ReactNode } from 'react';

import { Button, CloseButton } from './Button';
import { Dialog } from './Dialog';

/** Closes after confirmation even on error; callers must report failures. */
export const ConfirmDialog = ({
  body,
  confirmLabel,
  finalFocusEl,
  isDestructive = true,
  isConfirmDisabled = false,
  isOpen,
  onClose,
  onConfirm,
  onExitComplete,
  title,
}: {
  body: ReactNode;
  confirmLabel: string;
  /** Where focus returns on close when the opener may no longer exist, e.g. a row the confirmed action deleted. */
  finalFocusEl?: () => HTMLElement | null;
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
  const [isPending, setIsPending] = useState(false);
  const isPendingRef = useRef(false);
  const cancelRef = useRef<HTMLButtonElement>(null);
  // Hosts often clear the subject the dialog describes as it closes; keep the text it showed while it animates out.
  const live = useMemo(
    () => ({ body, confirmLabel, isDestructive, title }),
    [body, confirmLabel, isDestructive, title]
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

  // A destructive confirmation opens on Cancel rather than the dialog's default target (the close button for an
  // alertdialog), so a reflexive Enter or Space cancels and never commits or flips an option in the body.
  const getInitialFocus = useCallback(() => cancelRef.current, []);

  return (
    <Dialog.Root
      closeOnEscape={!isPending}
      closeOnInteractOutside={!isPending}
      finalFocusEl={finalFocusEl}
      initialFocusEl={text.isDestructive ? getInitialFocus : undefined}
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
            <Dialog.Body>
              <Stack gap="2">
                {typeof text.body === 'string' ? <Text fontSize="md">{text.body}</Text> : text.body}
              </Stack>
            </Dialog.Body>
            <Dialog.Footer>
              <Button ref={cancelRef} disabled={isPending} variant="ghost" onClick={handleClose}>
                Cancel
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
