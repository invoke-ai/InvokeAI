import { Dialog, Portal, Stack, Text } from '@chakra-ui/react';
import { useState, type ReactNode } from 'react';

import { Button, CloseButton } from './Button';

/**
 * Controlled confirmation dialog for consequential actions. The confirm
 * button shows a pending state while `onConfirm` runs; errors are left to the
 * caller (usually surfaced as a notification) and the dialog closes either way.
 */
export const ConfirmDialog = ({
  body,
  confirmLabel,
  isDestructive = true,
  isOpen,
  onClose,
  onConfirm,
  title,
}: {
  body: ReactNode;
  confirmLabel: string;
  isDestructive?: boolean;
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => Promise<void> | void;
  title: string;
}) => {
  const [isPending, setIsPending] = useState(false);

  const handleConfirm = async () => {
    setIsPending(true);

    try {
      await onConfirm();
    } finally {
      setIsPending(false);
      onClose();
    }
  };

  return (
    <Dialog.Root
      open={isOpen}
      placement="center"
      role="alertdialog"
      size="sm"
      onOpenChange={(event) => {
        if (!event.open) {
          onClose();
        }
      }}
    >
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content bg="bg.subtle" borderColor="border.subtle" borderWidth="1px" color="fg">
            <Dialog.Header>
              <Dialog.Title fontSize="sm" fontWeight="700">
                {title}
              </Dialog.Title>
            </Dialog.Header>
            <Dialog.Body>
              <Stack gap="2">{typeof body === 'string' ? <Text fontSize="xs">{body}</Text> : body}</Stack>
            </Dialog.Body>
            <Dialog.Footer gap="2">
              <Button disabled={isPending} size="xs" variant="ghost" onClick={onClose}>
                Cancel
              </Button>
              <Button
                colorPalette={isDestructive ? 'red' : 'accent'}
                loading={isPending}
                size="xs"
                variant="solid"
                onClick={() => {
                  void handleConfirm();
                }}
              >
                {confirmLabel}
              </Button>
            </Dialog.Footer>
            <Dialog.CloseTrigger asChild>
              <CloseButton color="fg.muted" size="sm" />
            </Dialog.CloseTrigger>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};
