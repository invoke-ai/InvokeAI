import { chakra, Input, Portal, Stack } from '@chakra-ui/react';
import { useExitRetainedValue } from '@platform/react/useExitRetainedValue';
import { useCallback, useMemo, useState, type FormEvent } from 'react';

import { Button, CloseButton } from './Button';
import { Dialog } from './Dialog';
import { Field } from './Field';

/** Submit only changed, nonempty names; async failures keep the dialog open and callers surface the error. */
export const RenameDialog = ({
  finalFocusEl,
  initialName,
  isOpen,
  label = 'Project name',
  onClose,
  onExitComplete,
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
  /** After the close animation; hosts that retain the dialog's subject release it here. */
  onExitComplete?: () => void;
  onSubmit: (name: string) => Promise<void> | void;
  submitLabel?: string;
  submitUnchanged?: boolean;
  title?: string;
}) => {
  const [isPending, setIsPending] = useState(false);
  // Keep the labels the dialog showed while it animates out, even if the host clears its subject on close.
  const live = useMemo(() => ({ label, submitLabel, title }), [label, submitLabel, title]);
  const shown = useExitRetainedValue(isOpen ? live : null);
  const text = shown.value ?? live;
  const { release } = shown;
  const handleExitComplete = useCallback(() => {
    release();
    onExitComplete?.();
  }, [onExitComplete, release]);

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
      onExitComplete={handleExitComplete}
      onOpenChange={handleOpenChange}
    >
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content>
            <chakra.form onSubmit={handleSubmit}>
              <Dialog.Header>
                <Dialog.Title>{text.title}</Dialog.Title>
              </Dialog.Header>
              <Dialog.Body>
                <Stack gap="2">
                  <Field label={text.label}>
                    <Input autoFocus defaultValue={initialName} name="renameValue" size="lg" />
                  </Field>
                </Stack>
              </Dialog.Body>
              <Dialog.Footer>
                <Button disabled={isPending} type="button" variant="ghost" onClick={onClose}>
                  Cancel
                </Button>
                <Button loading={isPending} type="submit" variant="solid">
                  {text.submitLabel}
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
