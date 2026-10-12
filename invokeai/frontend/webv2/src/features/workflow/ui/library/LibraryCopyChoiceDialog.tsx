import type { LibraryCopyChoiceRequest } from '@features/workflow/ui/workflowUiStore';

import { createListCollection, Portal, Stack } from '@chakra-ui/react';
import { useWorkflowProjectSelector, useWorkflowUi } from '@features/workflow/ui/WorkflowUiContext';
import { clearLibraryCopyChoice, setWorkflowLibraryOpen, workflowUiStore } from '@features/workflow/ui/workflowUiStore';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Button, CloseButton } from '@platform/ui/Button';
import { Dialog } from '@platform/ui/Dialog';
import { Field } from '@platform/ui/Field';
import { Select } from '@platform/ui/Select';
import { useCallback, useMemo, useRef, useState, type RefObject } from 'react';
import { useTranslation } from 'react-i18next';

import { findProjectCopiesOf } from './projectWorkflowEntries';
import { useOpenLibraryWorkflow, type OpenLibraryWorkflow } from './useOpenLibraryWorkflow';

/**
 * Asks what opening a template the project already holds should do; one host serves the library and the palette.
 * The root stays mounted and only its `open` changes: a dialog that mounts open while the menu that asked for it is
 * still closing gets dismissed with that menu's layer.
 */
export const LibraryCopyChoiceHost = () => {
  const { project } = useWorkflowUi();
  const request = workflowUiStore.useSelector((snapshot) => snapshot.libraryCopyChoice);
  const projectId = useWorkflowProjectSelector((snapshot) => snapshot.id);
  // A count, not the copies: the host stays mounted and must not re-render on every edit to the project's graphs.
  const copyCount = useWorkflowProjectSelector((snapshot) =>
    request ? findProjectCopiesOf(snapshot.workflows, request.item.workflow_id).length : 0
  );
  const openButtonRef = useRef<HTMLButtonElement | null>(null);
  const requestId = request?.requestId ?? null;
  const close = useCallback(() => {
    if (requestId !== null) {
      clearLibraryCopyChoice(requestId);
    }
  }, [requestId]);
  const finish = useCallback(() => {
    close();
    setWorkflowLibraryOpen(false);
  }, [close]);
  const loader = useOpenLibraryWorkflow(finish);
  const isBusy = loader.loadPhase !== 'idle';
  // About copies that are all gone, there is nothing left to ask.
  const isOpen = request !== null && request.projectId === projectId && copyCount > 0;
  const handleOpenChange = useCallback(
    (event: { open: boolean }) => {
      if (!event.open && !isBusy) {
        close();
      }
    },
    [close, isBusy]
  );
  // Opening is the safe default: Enter on arrival must never replace a copy.
  const getInitialFocus = useCallback(() => openButtonRef.current, []);

  /* eslint-disable react-hooks/rules-of-hooks -- useMountEffect is the repository's explicit useEffect wrapper */
  // A question about another project's copies ends when that project stops being active.
  useMountEffect(() =>
    project.subscribe(() => {
      const pending = workflowUiStore.getSnapshot().libraryCopyChoice;

      if (pending && pending.projectId !== project.getSnapshot().id) {
        clearLibraryCopyChoice(pending.requestId);
      }
    })
  );
  /* eslint-enable react-hooks/rules-of-hooks */

  return (
    <Dialog.Root
      closeOnEscape={!isBusy}
      closeOnInteractOutside={!isBusy}
      initialFocusEl={getInitialFocus}
      lazyMount
      open={isOpen}
      size="sm"
      unmountOnExit
      onOpenChange={handleOpenChange}
    >
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content data-library-copy-choice="">
            {isOpen ? (
              <LibraryCopyChoiceContent
                key={request.requestId}
                loader={loader}
                openButtonRef={openButtonRef}
                request={request}
              />
            ) : null}
            <Dialog.CloseTrigger asChild>
              <CloseButton disabled={isBusy} />
            </Dialog.CloseTrigger>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};

type PendingChoice = 'add' | 'replace' | null;

const LibraryCopyChoiceContent = ({
  loader: { loadPhase, open, replace, resume },
  openButtonRef,
  request,
}: {
  loader: OpenLibraryWorkflow;
  openButtonRef: RefObject<HTMLButtonElement | null>;
  request: LibraryCopyChoiceRequest;
}) => {
  const { t } = useTranslation();
  const workflows = useWorkflowProjectSelector((snapshot) => snapshot.workflows);
  const activeWorkflowId = useWorkflowProjectSelector((snapshot) => snapshot.activeWorkflowId);
  const copies = useMemo(
    () => findProjectCopiesOf(workflows, request.item.workflow_id),
    [request.item.workflow_id, workflows]
  );
  // The copy being edited is the likeliest one meant; otherwise the oldest.
  const [selectedCopyId, setSelectedCopyId] = useState(
    copies.some((copy) => copy.document.id === activeWorkflowId) ? activeWorkflowId : copies[0]!.document.id
  );
  const [pending, setPending] = useState<PendingChoice>(null);
  const copyId = copies.some((copy) => copy.document.id === selectedCopyId) ? selectedCopyId : copies[0]!.document.id;
  const selectValue = useMemo(() => [copyId], [copyId]);
  const isBusy = loadPhase !== 'idle';
  const name = request.item.name || t('workflowLibrary.untitled');
  const copyCollection = useMemo(
    () =>
      createListCollection({
        // Copies often share the template's name; the open one is told apart.
        items: copies.map((copy) => {
          const copyName = copy.document.name || t('workflowLibrary.untitled');

          return {
            label:
              copy.document.id === activeWorkflowId
                ? t('workflowLibrary.copyChoice.openCopyLabel', { name: copyName })
                : copyName,
            value: copy.document.id,
          };
        }),
      }),
    [activeWorkflowId, copies, t]
  );

  const handleCopyChange = useCallback(({ value }: { value: string[] }) => {
    if (value[0]) {
      setSelectedCopyId(value[0]);
    }
  }, []);
  const handleResume = useCallback(() => resume(copyId), [copyId, resume]);
  const handleAdd = useCallback(() => {
    setPending('add');
    void open(request.item, 'add-copy').finally(() => setPending(null));
  }, [open, request.item]);
  const handleReplace = useCallback(() => {
    setPending('replace');
    void replace(request.item, copyId).finally(() => setPending(null));
  }, [copyId, replace, request.item]);

  return (
    <>
      <Dialog.Header>
        <Dialog.Title>{t('workflowLibrary.copyChoice.title', { name })}</Dialog.Title>
      </Dialog.Header>
      <Dialog.Body>
        <Stack gap="3">
          <Dialog.Description fontSize="md">
            {t('workflowLibrary.copyChoice.body', { count: copies.length })}{' '}
            {t('workflowLibrary.copyChoice.replaceHint')}
          </Dialog.Description>
          {copies.length > 1 ? (
            <Field label={t('workflowLibrary.copyChoice.copy')}>
              <Select
                collection={copyCollection}
                portalled={false}
                value={selectValue}
                onValueChange={handleCopyChange}
              />
            </Field>
          ) : null}
        </Stack>
      </Dialog.Body>
      <Dialog.Footer>
        <Button
          colorPalette="red"
          disabled={isBusy}
          loading={pending === 'replace'}
          me="auto"
          variant="outline"
          onClick={handleReplace}
        >
          {t('workflowLibrary.copyChoice.replace')}
        </Button>
        <Button disabled={isBusy} loading={pending === 'add'} variant="outline" onClick={handleAdd}>
          {t('workflowLibrary.copyChoice.addCopy')}
        </Button>
        <Button ref={openButtonRef} disabled={isBusy} variant="solid" onClick={handleResume}>
          {t('workflowLibrary.copyChoice.open')}
        </Button>
      </Dialog.Footer>
    </>
  );
};
