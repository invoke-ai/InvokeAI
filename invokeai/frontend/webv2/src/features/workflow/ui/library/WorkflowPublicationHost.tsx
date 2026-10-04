import type { ProjectWorkflowEntry } from '@features/workflow/core/types';
import type { WorkflowRecordDTO } from '@features/workflow/queries';
import type { FormEvent } from 'react';

import { chakra, Dialog, Input, Portal, Stack, Text } from '@chakra-ui/react';
import { getLibraryWorkflowRecord } from '@features/workflow/queries';
import { useInvocationTemplatesSnapshot } from '@features/workflow/react';
import { useWorkflowProjectSelector, useWorkflowUi } from '@features/workflow/ui/WorkflowUiContext';
import { clearWorkflowPublicationIntent, workflowUiStore } from '@features/workflow/ui/workflowUiStore';
import { parseWorkflowJson } from '@features/workflow/utility';
import { useMountEffect } from '@platform/react/useMountEffect';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, CloseButton } from '@platform/ui/Button';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { Field } from '@platform/ui/Field';
import { Suspense, useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import {
  buildLibraryGraphPreviewSource,
  DeferredGraphPreviewDialog,
  preloadGraphPreview,
} from './libraryPreviewSource';
import { isUpdatableSource } from './projectWorkflowEntries';
import { formatRelativeTime } from './relativeTime';
import {
  discardUnresolvedWorkflowPublication,
  getUnresolvedWorkflowPublication,
  takeUnresolvedWorkflowPublicationOffer,
  useWorkflowPublication,
  type WorkflowPublicationFailure,
} from './useWorkflowPublication';

type FailedResult = WorkflowPublicationFailure;

/**
 * The publication dialogs, always mounted beside the editor so any surface (header, menu, the project view) can
 * start one for any project workflow. One stage at a time; every stage names the workflow it works on, and a
 * workflow that leaves the project ends its stage.
 */
/** Every stage names the project and workflow it belongs to and the operation that started it. */
interface StageContext {
  projectId: string;
  workflowId: string;
  operation: number;
  /** A publication is in flight for this stage; its dialogs stay open and their controls wait. */
  busy: boolean;
}

type ActiveStage = StageContext &
  (
    | { kind: 'save-as-new' }
    | { kind: 'confirm-update'; libraryWorkflowId: string; expectedRevision: number }
    | {
        kind: 'conflict';
        libraryWorkflowId: string;
        /** Null when the copy's revision was never known (a migrated project). */
        currentRevision: number | null;
      }
    | { kind: 'unavailable'; reason: 'missing' | 'forbidden' | 'bundled' }
    | {
        kind: 'review';
        libraryWorkflowId: string;
        record: WorkflowRecordDTO | null;
        error: string | null;
        isPreviewOpen: boolean;
      }
    | { kind: 'retry'; failed: FailedResult }
  );

type Stage = { kind: 'idle' } | ActiveStage;

const IDLE: Stage = { kind: 'idle' };

const contextOf = (stage: ActiveStage, busy = false): StageContext => ({
  busy,
  operation: stage.operation,
  projectId: stage.projectId,
  workflowId: stage.workflowId,
});

export const WorkflowPublicationHost = () => {
  const { t } = useTranslation();
  const { project } = useWorkflowUi();
  const workflows = useWorkflowProjectSelector((project) => project.workflows);
  const projectId = useWorkflowProjectSelector((project) => project.id);
  const templatesSnapshot = useInvocationTemplatesSnapshot();
  const publication = useWorkflowPublication();
  const [stage, setStage] = useState<Stage>(IDLE);
  const operationRef = useRef(0);

  const findEntry = useCallback(
    (workflowId: string): ProjectWorkflowEntry | undefined =>
      workflows.find((entry) => entry.document.id === workflowId),
    [workflows]
  );

  // An intent is consumed as it is posted: the stage it opens is read from the project as it stands then. Each
  // intent starts a new operation, and a project switch ends whatever stage was open, so a late answer from an
  // earlier operation can never land on a stage it did not start. A publication that never got an answer is
  // offered once more, under its own reserved id, when the user comes back to that workflow, and always meets a
  // new save for that workflow first.
  useMountEffect(() => {
    const retryStage = (failed: FailedResult, projectId: string, workflowId: string): Stage => {
      operationRef.current += 1;
      return { busy: false, failed, kind: 'retry', operation: operationRef.current, projectId, workflowId };
    };
    const consume = () => {
      const intent = workflowUiStore.getSnapshot().publicationIntent;

      if (!intent) {
        return;
      }

      clearWorkflowPublicationIntent();

      const snapshot = project.getSnapshot();
      const entry = snapshot.workflows.find((candidate) => candidate.document.id === intent.workflowId);

      if (!entry) {
        return;
      }

      // An unanswered save comes first: resending it (or discarding it knowingly) is how a duplicate is avoided.
      const unresolved = getUnresolvedWorkflowPublication(snapshot.id, intent.workflowId);

      if (unresolved) {
        setStage(retryStage(unresolved, snapshot.id, intent.workflowId));
        return;
      }

      operationRef.current += 1;
      const context: StageContext = {
        busy: false,
        operation: operationRef.current,
        projectId: snapshot.id,
        workflowId: intent.workflowId,
      };

      if (intent.kind === 'save-as-new') {
        setStage({ ...context, kind: 'save-as-new' });
      } else if (isUpdatableSource(entry.source)) {
        setStage(
          entry.source.revision === null
            ? { ...context, currentRevision: null, kind: 'conflict', libraryWorkflowId: entry.source.libraryWorkflowId }
            : {
                ...context,
                expectedRevision: entry.source.revision,
                kind: 'confirm-update',
                libraryWorkflowId: entry.source.libraryWorkflowId,
              }
        );
      } else if (entry.source) {
        setStage({ ...context, kind: 'unavailable', reason: 'bundled' });
      }
    };
    let lastActiveKey = '';
    const followActiveWorkflow = () => {
      const snapshot = project.getSnapshot();
      const activeKey = `${snapshot.id}\u0000${snapshot.activeWorkflowId}`;
      const arrived = activeKey !== lastActiveKey;

      lastActiveKey = activeKey;

      // The offer is taken on arrival, outside the updater, so it is made exactly once per return.
      const offered = arrived ? takeUnresolvedWorkflowPublicationOffer(snapshot.id, snapshot.activeWorkflowId) : null;
      const recovery = offered ? retryStage(offered, snapshot.id, snapshot.activeWorkflowId) : null;

      setStage((current) => {
        const kept = current.kind !== 'idle' && current.projectId !== snapshot.id ? IDLE : current;

        return recovery && kept.kind === 'idle' ? recovery : kept;
      });
    };

    consume();
    followActiveWorkflow();
    const unsubscribers = [workflowUiStore.subscribe(consume), project.subscribe(followActiveWorkflow)];

    return () => unsubscribers.forEach((unsubscribe) => unsubscribe());
  });

  const activeStage: Stage =
    stage.kind !== 'idle' && (stage.projectId !== projectId || !findEntry(stage.workflowId)) ? IDLE : stage;
  const isBusy = activeStage.kind !== 'idle' && activeStage.busy;
  const stageEntry = activeStage.kind === 'idle' ? undefined : findEntry(activeStage.workflowId);
  const workflowName = stageEntry?.document.name || t('workflowLibrary.untitled');

  /** Moves the stage on only while the operation that asked for the move is still the one on screen. */
  const advance = useCallback(
    (operation: number, next: (current: ActiveStage) => Stage) =>
      setStage((current) => (current.kind === 'idle' || current.operation !== operation ? current : next(current))),
    []
  );

  const close = useCallback(() => setStage(IDLE), []);

  /** Routes a settled publication to its follow-up stage; published, invalid and busy are reported by the hook. */
  const follow = useCallback(
    (operation: number, result: Awaited<ReturnType<typeof publication.saveAsNew>>) =>
      advance(operation, (current) => {
        const context = contextOf(current);

        switch (result.status) {
          case 'conflict':
            return {
              ...context,
              currentRevision: result.currentRevision,
              kind: 'conflict',
              libraryWorkflowId: result.libraryWorkflowId,
            };
          case 'unavailable':
            return { ...context, kind: 'unavailable', reason: result.reason };
          case 'failed':
            return { ...context, failed: result, kind: 'retry' };
          default:
            return IDLE;
        }
      }),
    [advance]
  );

  const beginBusy = useCallback(
    (operation: number) => advance(operation, (current) => ({ ...current, busy: true })),
    [advance]
  );

  const handleSaveAsNew = useCallback(
    async (name: string) => {
      if (activeStage.kind === 'idle') {
        return;
      }

      const { operation, workflowId } = activeStage;

      beginBusy(operation);
      follow(operation, await publication.saveAsNew({ name, workflowId }));
    },
    [activeStage, beginBusy, follow, publication]
  );

  const handleUpdate = useCallback(
    async (target: Extract<ActiveStage, { kind: 'confirm-update' }>) => {
      const { expectedRevision, libraryWorkflowId, operation, workflowId } = target;

      beginBusy(operation);
      follow(operation, await publication.updateSource({ expectedRevision, libraryWorkflowId, workflowId }));
    },
    [beginBusy, follow, publication]
  );

  const handleRetry = useCallback(async () => {
    if (activeStage.kind !== 'retry') {
      return;
    }

    const { failed, operation } = activeStage;

    beginBusy(operation);
    follow(operation, await publication.retry(failed));
  }, [activeStage, beginBusy, follow, publication]);

  const startReview = useCallback(async () => {
    if (activeStage.kind !== 'conflict') {
      return;
    }

    const { libraryWorkflowId, operation } = activeStage;
    const owner = captureAccountScope();

    advance(operation, (current) => ({
      ...contextOf(current),
      error: null,
      isPreviewOpen: false,
      kind: 'review',
      libraryWorkflowId,
      record: null,
    }));

    try {
      // Always the live record: the review is what the replacement's revision is taken from.
      const record = await getLibraryWorkflowRecord(libraryWorkflowId, owner.signal);

      assertAccountScopeCurrent(owner);
      advance(operation, (current) => (current.kind === 'review' ? { ...current, record } : current));
    } catch (error) {
      if (!isAccountScopeCurrent(owner)) {
        return;
      }

      advance(operation, (current) =>
        current.kind === 'review'
          ? { ...current, error: getApiErrorMessage(error, t('workflowLibrary.reviewFailed')) }
          : current
      );
    }
  }, [activeStage, advance, t]);

  const switchToSaveAsNew = useCallback(() => {
    if (activeStage.kind !== 'idle') {
      advance(activeStage.operation, (current) => ({ ...contextOf(current), kind: 'save-as-new' }));
    }
  }, [activeStage, advance]);

  const reviewPreviewSource = useMemo(() => {
    if (activeStage.kind !== 'review' || !activeStage.record || templatesSnapshot.status !== 'loaded') {
      return null;
    }

    try {
      const { document } = parseWorkflowJson({ ...activeStage.record.workflow, id: activeStage.record.workflow_id });

      return buildLibraryGraphPreviewSource(document, templatesSnapshot.templates);
    } catch {
      return null;
    }
  }, [activeStage, templatesSnapshot]);

  const openReviewPreview = useCallback(
    () => setStage((current) => (current.kind === 'review' ? { ...current, isPreviewOpen: true } : current)),
    []
  );
  const handleReviewPreviewOpenChange = useCallback(
    (open: boolean) =>
      setStage((current) => (current.kind === 'review' && !open ? { ...current, isPreviewOpen: false } : current)),
    []
  );

  const confirmReplaceReviewed = useCallback(() => {
    if (activeStage.kind === 'review' && activeStage.record) {
      const { libraryWorkflowId, operation, record } = activeStage;

      advance(operation, (current) => ({
        ...contextOf(current),
        expectedRevision: record.revision,
        kind: 'confirm-update',
        libraryWorkflowId,
      }));
    }
  }, [activeStage, advance]);
  // The confirm dialog closes itself once its confirmation settles; that close belongs to the operation the dialog
  // was shown for, so it neither ends the update it started nor a confirmation another operation opened since.
  const confirmOperation = activeStage.kind === 'confirm-update' ? activeStage.operation : null;
  // The confirm dialog is one instance per operation: an in-flight confirmation of one operation never leaves
  // another operation's confirmation disabled. The key follows the operation from the stage's first step, so the
  // instance is already mounted when a review turns into a confirmation, and it stays put through an ordinary close.
  const stageOperation = activeStage.kind === 'idle' ? null : activeStage.operation;
  const [confirmKey, setConfirmKey] = useState(0);

  if (stageOperation !== null && stageOperation !== confirmKey) {
    setConfirmKey(stageOperation);
  }

  const closeConfirmUpdate = useCallback(() => {
    setStage((current) =>
      current.kind === 'confirm-update' && !current.busy && current.operation === confirmOperation ? IDLE : current
    );
  }, [confirmOperation]);
  const confirmUpdate = useCallback(() => {
    if (activeStage.kind === 'confirm-update') {
      return handleUpdate(activeStage);
    }

    close();
    return undefined;
  }, [activeStage, close, handleUpdate]);
  const handleDialogOpenChange = useCallback((event: { open: boolean }) => (event.open ? undefined : close()), [close]);
  const handleReview = useCallback(() => void startReview(), [startReview]);
  const handleRetryClick = useCallback(() => void handleRetry(), [handleRetry]);
  // Discarding is a knowing choice: the dialog says the template may already exist if the library did receive it.
  const handleDiscard = useCallback(() => {
    if (activeStage.kind === 'retry') {
      discardUnresolvedWorkflowPublication(activeStage.projectId, activeStage.workflowId);
    }

    close();
  }, [activeStage, close]);

  const updateConfirmBody = useMemo(
    () => (
      <Stack gap="2">
        <Text>{t('workflowLibrary.updateConfirmBody', { name: workflowName })}</Text>
        <Text color="fg.muted" fontSize="md">
          {t('workflowLibrary.updateConfirmCallers')}
        </Text>
      </Stack>
    ),
    [t, workflowName]
  );
  const conflictOptions = useMemo(
    () => [
      { label: t('workflowLibrary.saveAsNew'), onSelect: switchToSaveAsNew, value: 'save-as-new' },
      { label: t('workflowLibrary.reviewTemplate'), onSelect: handleReview, value: 'review' },
    ],
    [handleReview, switchToSaveAsNew, t]
  );
  const saveAsNewOptions = useMemo(
    () => [{ label: t('workflowLibrary.saveAsNew'), onSelect: switchToSaveAsNew, value: 'save-as-new' }],
    [switchToSaveAsNew, t]
  );
  const retryOptions = useMemo(
    () => [
      { label: t('workflowLibrary.retryDiscard'), onSelect: handleDiscard, value: 'discard' },
      { label: t('workflowLibrary.retry'), onSelect: handleRetryClick, value: 'retry' },
    ],
    [handleDiscard, handleRetryClick, t]
  );

  return (
    <>
      <SaveToLibraryDialog
        initialName={stageEntry?.document.name ?? ''}
        isOpen={activeStage.kind === 'save-as-new'}
        isPending={isBusy}
        onClose={close}
        onSubmit={handleSaveAsNew}
      />

      <ConfirmDialog
        key={confirmKey}
        body={updateConfirmBody}
        confirmLabel={t('workflowLibrary.updateConfirm')}
        isDestructive={false}
        isOpen={activeStage.kind === 'confirm-update'}
        title={t('workflowLibrary.updateTitle')}
        onClose={closeConfirmUpdate}
        onConfirm={confirmUpdate}
      />

      <ChoiceDialog
        body={
          activeStage.kind === 'conflict' && activeStage.currentRevision === null
            ? t('workflowLibrary.conflictUnknownBody', { name: workflowName })
            : t('workflowLibrary.conflictStaleBody', { name: workflowName })
        }
        isOpen={activeStage.kind === 'conflict'}
        options={conflictOptions}
        title={t('workflowLibrary.conflictTitle')}
        onClose={close}
      />

      <ChoiceDialog
        body={
          activeStage.kind === 'unavailable'
            ? t(
                activeStage.reason === 'missing'
                  ? 'workflowLibrary.unavailableMissingBody'
                  : activeStage.reason === 'bundled'
                    ? 'workflowLibrary.unavailableBundledBody'
                    : 'workflowLibrary.unavailableForbiddenBody',
                { name: workflowName }
              )
            : ''
        }
        isOpen={activeStage.kind === 'unavailable'}
        options={saveAsNewOptions}
        title={t('workflowLibrary.unavailableTitle')}
        onClose={close}
      />

      <ChoiceDialog
        body={
          activeStage.kind === 'retry'
            ? t('workflowLibrary.retryBody', { message: activeStage.failed.message, name: workflowName })
            : ''
        }
        isBusy={isBusy}
        isOpen={activeStage.kind === 'retry'}
        options={retryOptions}
        title={t('workflowLibrary.retryTitle')}
        onClose={close}
      />

      <Dialog.Root
        lazyMount
        open={activeStage.kind === 'review'}
        size="sm"
        unmountOnExit
        onOpenChange={handleDialogOpenChange}
      >
        <Portal>
          <Dialog.Backdrop />
          <Dialog.Positioner>
            <Dialog.Content ref={preloadGraphPreview} data-workflow-review-dialog>
              <Dialog.Header>
                <Dialog.Title>{t('workflowLibrary.reviewTitle')}</Dialog.Title>
              </Dialog.Header>
              <Dialog.Body>
                {activeStage.kind === 'review' ? (
                  <ReviewBody
                    error={activeStage.error}
                    record={activeStage.record}
                    workflowName={workflowName}
                    onPreview={reviewPreviewSource ? openReviewPreview : undefined}
                  />
                ) : null}
              </Dialog.Body>
              <Dialog.Footer>
                <Button variant="ghost" onClick={close}>
                  {t('common.cancel')}
                </Button>
                <Button variant="outline" onClick={switchToSaveAsNew}>
                  {t('workflowLibrary.saveAsNew')}
                </Button>
                <Button
                  disabled={activeStage.kind !== 'review' || !activeStage.record}
                  variant="solid"
                  onClick={confirmReplaceReviewed}
                >
                  {t('workflowLibrary.replaceTemplate')}
                </Button>
              </Dialog.Footer>
              <Dialog.CloseTrigger asChild>
                <CloseButton />
              </Dialog.CloseTrigger>
            </Dialog.Content>
          </Dialog.Positioner>
        </Portal>
      </Dialog.Root>

      {activeStage.kind === 'review' && activeStage.record && reviewPreviewSource ? (
        <Suspense fallback={null}>
          <DeferredGraphPreviewDialog
            graphId={activeStage.record.workflow_id}
            hideInvoke
            isOpen={activeStage.isPreviewOpen}
            source={reviewPreviewSource}
            sourceLabel={activeStage.record.name || t('workflowLibrary.untitled')}
            onOpenChange={handleReviewPreviewOpenChange}
          />
        </Suspense>
      ) : null}
    </>
  );
};

const ReviewBody = ({
  error,
  record,
  workflowName,
  onPreview,
}: {
  error: string | null;
  record: WorkflowRecordDTO | null;
  workflowName: string;
  onPreview: (() => void) | undefined;
}) => {
  const { t } = useTranslation();

  if (error) {
    return (
      <Text color="fg.error" fontSize="lg">
        {error}
      </Text>
    );
  }

  if (!record) {
    return (
      <Text color="fg.subtle" fontSize="lg" role="status">
        {t('workflowLibrary.reviewLoading')}
      </Text>
    );
  }

  return (
    <Stack gap="2">
      <Text fontSize="lg">{t('workflowLibrary.reviewBody', { name: workflowName })}</Text>
      <Stack gap="0.5">
        <Text fontSize="lg" fontWeight="600" overflowWrap="anywhere">
          {record.name || t('workflowLibrary.untitled')}
        </Text>
        {record.description ? (
          <Text color="fg.muted" fontSize="md" lineClamp={3}>
            {record.description}
          </Text>
        ) : null}
        <Text color="fg.subtle" fontSize="xs">
          {t('workflowLibrary.reviewRevision', {
            revision: record.revision,
            when: record.updated_at ? formatRelativeTime(record.updated_at, new Date()) : '',
          })}
        </Text>
      </Stack>
      {onPreview ? (
        <Button alignSelf="flex-start" variant="outline" onClick={onPreview}>
          {t('workflowLibrary.previewGraph')}
        </Button>
      ) : null}
    </Stack>
  );
};

/** A short decision with named choices; Escape and the close control cancel. */
const ChoiceDialog = ({
  body,
  isBusy = false,
  isOpen,
  options,
  title,
  onClose,
}: {
  body: string;
  isBusy?: boolean;
  isOpen: boolean;
  options: ReadonlyArray<{ label: string; onSelect: () => void; value: string }>;
  title: string;
  onClose: () => void;
}) => {
  const { t } = useTranslation();
  const handleOpenChange = useCallback((event: { open: boolean }) => (event.open ? undefined : onClose()), [onClose]);

  return (
    <Dialog.Root
      closeOnEscape={!isBusy}
      closeOnInteractOutside={!isBusy}
      lazyMount
      open={isOpen}
      size="xs"
      unmountOnExit
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
              <Text fontSize="lg">{body}</Text>
            </Dialog.Body>
            <Dialog.Footer>
              <Button disabled={isBusy} variant="ghost" onClick={onClose}>
                {t('common.cancel')}
              </Button>
              {options.map((option) => (
                <ChoiceButton key={option.value} isBusy={isBusy} option={option} />
              ))}
            </Dialog.Footer>
            <Dialog.CloseTrigger asChild>
              <CloseButton disabled={isBusy} />
            </Dialog.CloseTrigger>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};

const ChoiceButton = ({
  isBusy,
  option,
}: {
  isBusy: boolean;
  option: { label: string; onSelect: () => void; value: string };
}) => (
  <Button data-choice={option.value} loading={isBusy} variant="solid" onClick={option.onSelect}>
    {option.label}
  </Button>
);

/** Names the new template. The explanation is the contract: current input values become the template's defaults. */
export const SaveToLibraryDialog = ({
  initialName,
  isOpen,
  isPending,
  onClose,
  onSubmit,
}: {
  initialName: string;
  isOpen: boolean;
  isPending: boolean;
  onClose: () => void;
  onSubmit: (name: string) => Promise<void> | void;
}) => {
  const { t } = useTranslation();

  const handleSubmit = useCallback(
    (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      const name = new FormData(event.currentTarget).get('templateName')?.toString().trim() ?? '';

      void onSubmit(name || t('workflowLibrary.untitled'));
    },
    [onSubmit, t]
  );
  const handleOpenChange = useCallback((event: { open: boolean }) => (event.open ? undefined : onClose()), [onClose]);

  return (
    <Dialog.Root
      closeOnEscape={!isPending}
      closeOnInteractOutside={!isPending}
      lazyMount
      open={isOpen}
      size="xs"
      unmountOnExit
      onOpenChange={handleOpenChange}
    >
      <Portal>
        <Dialog.Backdrop />
        <Dialog.Positioner>
          <Dialog.Content data-save-to-library-dialog>
            <chakra.form onSubmit={handleSubmit}>
              <Dialog.Header>
                <Dialog.Title>{t('workflowLibrary.saveToLibraryTitle')}</Dialog.Title>
              </Dialog.Header>
              <Dialog.Body>
                <Stack gap="3">
                  <Field label={t('workflowLibrary.templateName')}>
                    <Input defaultValue={initialName} name="templateName" size="lg" />
                  </Field>
                  <Text color="fg.muted" fontSize="md">
                    {t('workflowLibrary.saveToLibraryExplanation')}
                  </Text>
                </Stack>
              </Dialog.Body>
              <Dialog.Footer>
                <Button disabled={isPending} type="button" variant="ghost" onClick={onClose}>
                  {t('common.cancel')}
                </Button>
                <Button loading={isPending} type="submit" variant="solid">
                  {t('workflowLibrary.saveToLibraryConfirm')}
                </Button>
              </Dialog.Footer>
            </chakra.form>
            <Dialog.CloseTrigger asChild>
              <CloseButton disabled={isPending} />
            </Dialog.CloseTrigger>
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};
