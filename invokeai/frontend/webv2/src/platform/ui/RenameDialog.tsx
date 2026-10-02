import { chakra, Dialog, Input, Portal, Stack } from '@chakra-ui/react';
import { useCallback, useState, type FormEvent } from 'react';

import { Button, CloseButton } from './Button';
import { Field } from './Field';

/** Submit only changed, nonempty names; async failures keep the dialog open and callers surface the error. */
export const RenameDialog = ({
  finalFocusEl,
  initialName,
  isOpen,
  label = 'Project name',
  onClose,
  onSubmit,
  submitLabel = 'Rename',
  submitUnchanged = false,
  title = 'Rename project',
}: {
  finalFocusEl?: () => HTMLElement | null;
  initialName: string;
  isOpen: boolean;
  label?: string;
  onClose: () => void;
  onSubmit: (name: string) => Promise<void> | void;
  submitLabel?: string;
  submitUnchanged?: boolean;
  title?: string;
}) => {
  const [isPending, setIsPending] = useState(false);

  const commit = useCallback(
    async (value: string) => {
      const name = value.trim();

      if (!name || (!submitUnchanged && name === initialName.trim())) {
        onClose();

        return;
      }

      setIsPending(true);

      try {
        await onSubmit(name);
        onClose();
      } catch {
        // Keep the entered name after failure; the caller reports the error.
      } finally {
        setIsPending(false);
      }
    },
    [initialName, onClose, onSubmit, submitUnchanged]
  );

  const handleOpenChange = useCallback(
    (event: { open: boolean }) => {
      if (!event.open) {
        onClose();
      }
    },
    [onClose]
  );

  const handleSubmit = useCallback(
    (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      void commit(new FormData(event.currentTarget).get('renameValue')?.toString() ?? '');
    },
    [commit]
  );

  return (
    <Dialog.Root
      lazyMount
      finalFocusEl={finalFocusEl}
      open={isOpen}
      size="xs"
      unmountOnExit
      onOpenChange={handleOpenChange}
    >
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content>
            <chakra.form onSubmit={handleSubmit}>
              <Dialog.Header>
                <Dialog.Title>{title}</Dialog.Title>
              </Dialog.Header>
              <Dialog.Body>
                <Stack gap="2">
                  <Field label={label}>
                    <Input autoFocus defaultValue={initialName} name="renameValue" size="sm" />
                  </Field>
                </Stack>
              </Dialog.Body>
              <Dialog.Footer>
                <Button disabled={isPending} size="xs" type="button" variant="ghost" onClick={onClose}>
                  Cancel
                </Button>
                <Button loading={isPending} size="xs" type="submit" variant="solid">
                  {submitLabel}
                </Button>
              </Dialog.Footer>
            </chakra.form>
            <Dialog.CloseTrigger asChild>
              <CloseButton />
            </Dialog.CloseTrigger>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};
