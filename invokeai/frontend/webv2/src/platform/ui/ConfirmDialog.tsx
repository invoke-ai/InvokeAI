import { Dialog, Portal, Stack, Text } from '@chakra-ui/react';
import { useCallback, useRef, useState, type ReactNode } from 'react';

import { Button, CloseButton } from './Button';

/** Closes after confirmation even on error; callers must report failures. */
export const ConfirmDialog = ({
  body,
  confirmLabel,
  finalFocusEl,
  isDestructive = true,
  isOpen,
  onClose,
  onConfirm,
  title,
}: {
  body: ReactNode;
  confirmLabel: string;
  /** Where focus returns on close when the opener may no longer exist, e.g. a row the confirmed action deleted. */
  finalFocusEl?: () => HTMLElement | null;
  isDestructive?: boolean;
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => Promise<void> | void;
  title: string;
}) => {
  const [isPending, setIsPending] = useState(false);
  const isPendingRef = useRef(false);

  const handleConfirm = useCallback(async () => {
    if (isPendingRef.current) {
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
  }, [onClose, onConfirm]);

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

  return (
    <Dialog.Root
      closeOnEscape={!isPending}
      closeOnInteractOutside={!isPending}
      finalFocusEl={finalFocusEl}
      open={isOpen}
      role="alertdialog"
      size="sm"
      onOpenChange={handleOpenChange}
    >
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content>
            <Dialog.Header>
              <Dialog.Title>{title}</Dialog.Title>
            </Dialog.Header>
            <Dialog.Body>
              <Stack gap="2">{typeof body === 'string' ? <Text fontSize="xs">{body}</Text> : body}</Stack>
            </Dialog.Body>
            <Dialog.Footer>
              <Button disabled={isPending} size="xs" variant="ghost" onClick={handleClose}>
                Cancel
              </Button>
              <Button
                colorPalette={isDestructive ? 'red' : 'accent'}
                disabled={isPending}
                loading={isPending}
                size="xs"
                variant="solid"
                onClick={handleConfirmClick}
              >
                {confirmLabel}
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
