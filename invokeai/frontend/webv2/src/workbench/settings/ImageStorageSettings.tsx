import type { ImageMoveStatus, ImageStorageLayout } from '@features/gallery/storage';

import { createListCollection, HStack, Spinner, Stack, Text } from '@chakra-ui/react';
import {
  IMAGE_STORAGE_LAYOUTS,
  imageMoveStatusQueryOptions,
  imageStorageLayoutQueryOptions,
  isImageStorageLayout,
  startImageMove,
  startImageMoveRecovery,
  updateImageStorageLayout,
} from '@features/gallery/storage';
import { useCapabilities } from '@features/identity';
import { useScopedAction } from '@platform/react/useScopedAction';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button } from '@platform/ui/Button';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { Field } from '@platform/ui/Field';
import { Select } from '@platform/ui/Select';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { type MouseEvent, useCallback, useId, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

const tKey = (key: string): string => `settings.imageStorage.${key}`;
const CONTROL_WIDTH = { base: 'full', md: '72' } as const;
const SELECT_POSITIONING = { strategy: 'fixed', sameWidth: true, hideWhenDetached: true } as const;

type ImageMoveAction = 'start' | 'recover';

/**
 * The job state a refusal was answered against. A running move journals one job per batch, so its job ids advance
 * (with no active job between batches) while it runs: running is one state however far it has got. Otherwise the
 * active and latest jobs tell the states apart. A refusal stops applying once the status moves past it (the move
 * that refused the start ends, for example).
 */
const getJobKey = (status: ImageMoveStatus | undefined): string => {
  if (!status) {
    return 'unknown';
  }

  return status.isRunning ? 'running' : `idle:${String(status.activeJobId)}:${String(status.latestJob?.id ?? null)}`;
};

interface JobAvailability {
  canRecover: boolean;
  canStart: boolean;
  /** Visible reason for the disabled buttons, unless the status line already explains it. */
  reasonKey: string | null;
}

/**
 * Each button is enabled only when the server would do something with it: Recover only for an interrupted job (any
 * other recovery is a no-op behind the server-wide maintenance lock), Start Move only when images need moving.
 */
const getJobAvailability = ({
  isActing,
  isSavingLayout,
  status,
}: {
  isActing: boolean;
  isSavingLayout: boolean;
  /** Null until a usable status has loaded. */
  status: ImageMoveStatus | null;
}): JobAvailability => {
  if (status === null) {
    return { canRecover: false, canStart: false, reasonKey: 'unavailableStatus' };
  }
  if (isActing) {
    // The confirmation dialog shows the request in flight.
    return { canRecover: false, canStart: false, reasonKey: null };
  }
  if (status.isRunning) {
    return { canRecover: false, canStart: false, reasonKey: 'unavailableRunning' };
  }
  if (isSavingLayout) {
    // The new layout changes what needs moving.
    return { canRecover: false, canStart: false, reasonKey: 'unavailableSavingLayout' };
  }
  if (status.activeJobId !== null) {
    // The status line tells the admin to run Recover before another move.
    return { canRecover: true, canStart: false, reasonKey: null };
  }
  if (status.needsMoveCount > 0) {
    return { canRecover: false, canStart: true, reasonKey: 'recoverNotNeeded' };
  }

  return { canRecover: false, canStart: false, reasonKey: 'nothingToDo' };
};

/** Server image storage: the folder layout for new images and the move/recovery job for existing ones. */
export const ImageStorageSettings = () => {
  const { t } = useTranslation();
  const { canManageAppConfig } = useCapabilities();

  if (!canManageAppConfig) {
    return (
      <Text color="fg.muted" fontSize="md">
        {t(tKey('adminOnly'))}
      </Text>
    );
  }

  return <ImageStorageEditor />;
};

const ImageStorageEditor = () => {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  // Both routes are admin-only; the editor only mounts for accounts that can read them.
  const layoutQuery = useQuery(imageStorageLayoutQueryOptions());
  // Polling lives with this editor: the gallery refresh at the end of a move needs it open until the job stops.
  const statusQuery = useQuery(imageMoveStatusQueryOptions());
  const actionErrorId = useId();
  const availabilityId = useId();
  const maintenanceHeadingId = useId();
  const maintenanceRef = useRef<HTMLDivElement>(null);
  // The button that opened the confirmation, and whether it was confirmed rather than dismissed.
  const openerRef = useRef<HTMLButtonElement | null>(null);
  const isConfirmedRef = useRef(false);
  // Cancel returns focus to the opener. A confirmed action reports its outcome in the maintenance group, and an
  // accepted job disables the opener, so focus lands on that group rather than the document body.
  const getFinalFocus = useCallback(() => {
    const opener = isConfirmedRef.current ? null : openerRef.current;

    return opener && !opener.disabled ? opener : maintenanceRef.current;
  }, []);
  const { isBusy: isSavingLayout, run: runLayoutSave } = useScopedAction();
  const { isBusy: isActing, run: runAction } = useScopedAction();
  const [layoutError, setLayoutError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<{ action: ImageMoveAction; jobKey: string; message: string } | null>(
    null
  );
  const [pendingAction, setPendingAction] = useState<ImageMoveAction | null>(null);

  const layout = layoutQuery.data ?? 'flat';
  const layoutCollection = useMemo(() => {
    const items = IMAGE_STORAGE_LAYOUTS.map((value) => ({ label: t(tKey(`layouts.${value}`)), value }));

    // A layout this build does not know stays selectable as itself rather than silently reading as flat.
    if (!isImageStorageLayout(layout)) {
      items.push({ label: t(tKey('layouts.unknown'), { layout }), value: layout as ImageStorageLayout });
    }

    return createListCollection({ items });
  }, [layout, t]);
  const layoutValue = useMemo(() => [layout], [layout]);

  const handleLayoutChange = useCallback(
    ({ value }: { value: string[] }) => {
      const next = value[0];

      if (!isImageStorageLayout(next) || next === layout) {
        return;
      }

      setLayoutError(null);
      void runLayoutSave(
        (owner) => updateImageStorageLayout(queryClient, next, owner),
        (_message, error) => setLayoutError(getApiErrorMessage(error, t(tKey('layoutSaveFailed'))))
      );
    },
    [layout, queryClient, runLayoutSave, t]
  );

  const requestAction = useCallback((action: ImageMoveAction, opener: HTMLButtonElement) => {
    openerRef.current = opener;
    isConfirmedRef.current = false;
    setPendingAction(action);
  }, []);
  const requestStart = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => requestAction('start', event.currentTarget),
    [requestAction]
  );
  const requestRecover = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => requestAction('recover', event.currentTarget),
    [requestAction]
  );
  const closeConfirmation = useCallback(() => setPendingAction(null), []);
  // Returns the request so the dialog shows it in flight; the dialog closes when it settles, either way.
  const confirmAction = useCallback(async () => {
    const action = pendingAction;

    if (action === null) {
      return;
    }

    isConfirmedRef.current = true;
    setActionError(null);
    await runAction(
      async (owner) => {
        await (action === 'start' ? startImageMove(queryClient, owner) : startImageMoveRecovery(queryClient, owner));
      },
      (_message, error) =>
        setActionError({
          action,
          // The refusal re-read the status before it settled, so the cache holds the job it was refused against.
          jobKey: getJobKey(queryClient.getQueryData(imageMoveStatusQueryOptions().queryKey)),
          message: getApiErrorMessage(error, t(tKey(action === 'start' ? 'startFailed' : 'recoverFailed'))),
        })
    );
  }, [pendingAction, queryClient, runAction, t]);
  const retryStatus = useCallback(() => void statusQuery.refetch(), [statusQuery]);
  const retryLayout = useCallback(() => void layoutQuery.refetch(), [layoutQuery]);

  const status = statusQuery.data ?? null;
  const shownActionError = actionError?.jobKey === getJobKey(statusQuery.data) ? actionError : null;
  const usableStatus = statusQuery.isError ? null : status;
  const availability = getJobAvailability({ isActing, isSavingLayout, status: usableStatus });
  // The move reads the layout per batch, so changing it while a job runs (or may be running: status unknown, or a
  // start still being accepted) would leave the images split between two layouts.
  const isLayoutLocked =
    layoutQuery.isPending || isSavingLayout || isActing || usableStatus === null || usableStatus.isRunning;
  // Each button references only what explains it: its own action's failure, and the note while it is disabled.
  const describedBy = (action: ImageMoveAction, isEnabled: boolean) =>
    [
      shownActionError?.action === action ? actionErrorId : null,
      availability.reasonKey && !isEnabled ? availabilityId : null,
    ]
      .filter(Boolean)
      .join(' ') || undefined;

  return (
    <Stack gap="4">
      <Field
        disabled={isLayoutLocked}
        error={layoutError}
        helpText={t(tKey('layoutHelp'))}
        label={t(tKey('layoutLabel'))}
      >
        {layoutQuery.isError ? (
          <HStack gap="2">
            <Text color="fg.error" fontSize="md" minW="0" role="alert" wordBreak="break-word">
              {getApiErrorMessage(layoutQuery.error, t(tKey('layoutLoadFailed')))}
            </Text>
            <Button size="sm" variant="outline" onClick={retryLayout}>
              {t('common.retry')}
            </Button>
          </HStack>
        ) : (
          <Select
            collection={layoutCollection}
            disabled={isLayoutLocked}
            portalled={false}
            positioning={SELECT_POSITIONING}
            size="lg"
            value={layoutValue}
            w={CONTROL_WIDTH}
            onValueChange={handleLayoutChange}
          />
        )}
      </Field>
      <Stack ref={maintenanceRef} aria-labelledby={maintenanceHeadingId} gap="2" role="group" tabIndex={-1}>
        <Text color="fg" fontSize="lg" fontWeight="500" id={maintenanceHeadingId}>
          {t(tKey('maintenanceHeading'))}
        </Text>
        <Text color="fg.muted" fontSize="md">
          {t(tKey('maintenanceDescription'))}
        </Text>
        <ImageMoveStatusLine
          isPending={statusQuery.isPending}
          loadError={statusQuery.isError ? getApiErrorMessage(statusQuery.error, t(tKey('statusLoadFailed'))) : null}
          status={status}
          onRetry={retryStatus}
        />
        <HStack gap="2" wrap="wrap">
          <Button
            aria-describedby={describedBy('start', availability.canStart)}
            disabled={!availability.canStart}
            loading={isActing && pendingAction === 'start'}
            size="lg"
            variant="outline"
            onClick={requestStart}
          >
            {t(tKey('startMove'))}
          </Button>
          <Button
            aria-describedby={describedBy('recover', availability.canRecover)}
            disabled={!availability.canRecover}
            loading={isActing && pendingAction === 'recover'}
            size="lg"
            variant="outline"
            onClick={requestRecover}
          >
            {t(tKey('recover'))}
          </Button>
        </HStack>
        {availability.reasonKey ? (
          <Text color="fg.muted" fontSize="xs" id={availabilityId}>
            {t(tKey(availability.reasonKey))}
          </Text>
        ) : null}
        {shownActionError ? (
          <Text color="fg.error" fontSize="md" id={actionErrorId} minW="0" role="alert" wordBreak="break-word">
            {shownActionError.message}
          </Text>
        ) : null}
      </Stack>
      <ConfirmDialog
        body={t(tKey(pendingAction === 'recover' ? 'recoverConfirmBody' : 'startConfirmBody'))}
        confirmLabel={t(tKey(pendingAction === 'recover' ? 'recover' : 'startMove'))}
        finalFocusEl={getFinalFocus}
        isDestructive={false}
        isOpen={pendingAction !== null}
        title={t(tKey(pendingAction === 'recover' ? 'recoverConfirmTitle' : 'startConfirmTitle'))}
        onClose={closeConfirmation}
        onConfirm={confirmAction}
      />
    </Stack>
  );
};

const ImageMoveStatusLine = ({
  isPending,
  loadError,
  onRetry,
  status,
}: {
  isPending: boolean;
  loadError: string | null;
  onRetry: () => void;
  status: ImageMoveStatus | null;
}) => {
  const { t } = useTranslation();

  if (loadError) {
    return (
      <HStack gap="2">
        <Text color="fg.error" fontSize="md" minW="0" role="alert" wordBreak="break-word">
          {loadError}
        </Text>
        <Button size="sm" variant="outline" onClick={onRetry}>
          {t('common.retry')}
        </Button>
      </HStack>
    );
  }

  if (isPending || status === null) {
    return <Spinner size="md" />;
  }

  const errorMessage = status.latestJob?.errorMessage || status.lastError;
  let summary: string;

  if (status.activeJobId !== null && !status.isRunning) {
    summary = t(tKey('statusNeedsRecovery'));
  } else if (status.isRunning) {
    summary = t(tKey(status.operation === 'recovery' ? 'statusRecovering' : 'statusMoving'));
  } else {
    summary = t(tKey('statusIdle'), { count: status.needsMoveCount });
  }

  return (
    <Stack gap="1" aria-live="polite" role="status">
      <HStack gap="2">
        {status.isRunning ? <Spinner size="md" /> : null}
        <Text color="fg" fontSize="md">
          {summary}
        </Text>
      </HStack>
      {status.latestJob && !status.isRunning && status.activeJobId === null ? (
        <Text color="fg.muted" fontSize="xs">
          {t(tKey('latestJob'), { state: t(tKey(`jobStates.${status.latestJob.state}`)) })}
        </Text>
      ) : null}
      {errorMessage ? (
        <Text color="fg.error" fontSize="xs" minW="0" whiteSpace="pre-wrap" wordBreak="break-word">
          {errorMessage}
        </Text>
      ) : null}
    </Stack>
  );
};
