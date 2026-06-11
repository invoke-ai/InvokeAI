import { chakra, Dialog, Input, Portal, Stack } from '@chakra-ui/react';
import { useState } from 'react';

import { Button, CloseButton } from './Button';
import { Field } from './Field';

/**
 * Controlled single-field rename dialog. `onSubmit` only fires for a
 * non-empty name that actually changed; it may be async, in which case the
 * submit button shows a pending state and errors keep the dialog open so the
 * caller's message (usually a toast) can be acted on.
 */
export const RenameDialog = ({
  initialName,
  isOpen,
  label = 'Project name',
  onClose,
  onSubmit,
  title = 'Rename project',
}: {
  initialName: string;
  isOpen: boolean;
  label?: string;
  onClose: () => void;
  onSubmit: (name: string) => Promise<void> | void;
  title?: string;
}) => {
  const [isPending, setIsPending] = useState(false);

  const commit = async (value: string) => {
    const name = value.trim();

    if (!name || name === initialName.trim()) {
      onClose();

      return;
    }

    setIsPending(true);

    try {
      await onSubmit(name);
      onClose();
    } catch {
      // The caller surfaced the failure (toast/notification); stay open so
      // the name is not lost.
    } finally {
      setIsPending(false);
    }
  };

  return (
    <Dialog.Root
      lazyMount
      open={isOpen}
      placement="center"
      size="xs"
      unmountOnExit
      onOpenChange={(event) => {
        if (!event.open) {
          onClose();
        }
      }}
    >
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content bg="bg.surface" borderColor="border.subtle" borderWidth="1px" color="fg.default">
            <chakra.form
              onSubmit={(event) => {
                event.preventDefault();
                void commit(new FormData(event.currentTarget).get('renameValue')?.toString() ?? '');
              }}
            >
              <Dialog.Header>
                <Dialog.Title fontSize="sm" fontWeight="700">
                  {title}
                </Dialog.Title>
              </Dialog.Header>
              <Dialog.Body>
                <Stack gap="2">
                  <Field label={label}>
                    <Input autoFocus defaultValue={initialName} name="renameValue" size="sm" />
                  </Field>
                </Stack>
              </Dialog.Body>
              <Dialog.Footer gap="2">
                <Button disabled={isPending} size="xs" type="button" variant="ghost" onClick={onClose}>
                  Cancel
                </Button>
                <Button loading={isPending} size="xs" type="submit" variant="solid">
                  Rename
                </Button>
              </Dialog.Footer>
            </chakra.form>
            <Dialog.CloseTrigger asChild>
              <CloseButton color="fg.muted" size="sm" />
            </Dialog.CloseTrigger>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};
