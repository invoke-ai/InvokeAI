import type { ImageMoveStatus, ImageStorageLayout } from '@features/gallery/queries';

import { createListCollection, HStack, Spinner, Stack, Text } from '@chakra-ui/react';
import {
  IMAGE_STORAGE_LAYOUTS,
  imageMoveStatusQueryOptions,
  imageStorageLayoutQueryOptions,
  isImageStorageLayout,
  startImageMove,
  startImageMoveRecovery,
  updateImageStorageLayout,
} from '@features/gallery/queries';
import { useCapabilities } from '@features/identity';
import { useScopedAction } from '@platform/react/useScopedAction';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button } from '@platform/ui/Button';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { Field } from '@platform/ui/Field';
import { Select } from '@platform/ui/Select';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

const tKey = (key: string): string => `settings.imageStorage.${key}`;
const CONTROL_WIDTH = { base: 'full', md: '72' } as const;
const SELECT_POSITIONING = { strategy: 'fixed', sameWidth: true, hideWhenDetached: true } as const;

type ImageMoveAction = 'start' | 'recover';

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
  const statusQuery = useQuery(imageMoveStatusQueryOptions());
  const { isBusy: isSavingLayout, run: runLayoutSave } = useScopedAction();
  const { isBusy: isActing, run: runAction } = useScopedAction();
  const [layoutError, setLayoutError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
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

  const requestStart = useCallback(() => setPendingAction('start'), []);
  const requestRecover = useCallback(() => setPendingAction('recover'), []);
  const closeConfirmation = useCallback(() => setPendingAction(null), []);
  const confirmAction = useCallback(() => {
    const action = pendingAction;

    if (action === null) {
      return;
    }

    setActionError(null);
    void runAction(
      async (owner) => {
        await (action === 'start' ? startImageMove(queryClient, owner) : startImageMoveRecovery(queryClient, owner));
      },
      (_message, error) =>
        setActionError(getApiErrorMessage(error, t(tKey(action === 'start' ? 'startFailed' : 'recoverFailed'))))
    );
  }, [pendingAction, queryClient, runAction, t]);
  const retryStatus = useCallback(() => void statusQuery.refetch(), [statusQuery]);
  const retryLayout = useCallback(() => void layoutQuery.refetch(), [layoutQuery]);

  const status = statusQuery.data ?? null;
  const isRunning = status?.isRunning ?? false;
  const hasActiveJob = status !== null && status.activeJobId !== null;
  const isStatusUsable = status !== null && !statusQuery.isError;
  const isBusy = isRunning || isActing || !isStatusUsable;

  return (
    <Stack gap="4">
      <Field
        disabled={layoutQuery.isPending || isSavingLayout || isRunning}
        error={layoutError}
        helpText={t(tKey('layoutHelp'))}
        label={t(tKey('layoutLabel'))}
      >
        {layoutQuery.isError ? (
          <HStack gap="2">
            <Text color="fg.error" fontSize="md" role="alert">
              {getApiErrorMessage(layoutQuery.error, t(tKey('layoutLoadFailed')))}
            </Text>
            <Button size="sm" variant="outline" onClick={retryLayout}>
              {t('common.retry')}
            </Button>
          </HStack>
        ) : (
          <Select
            collection={layoutCollection}
            disabled={layoutQuery.isPending || isSavingLayout || isRunning}
            portalled={false}
            positioning={SELECT_POSITIONING}
            size="lg"
            value={layoutValue}
            w={CONTROL_WIDTH}
            onValueChange={handleLayoutChange}
          />
        )}
      </Field>
      <Stack gap="2">
        <Text color="fg" fontSize="lg" fontWeight="500">
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
            disabled={isBusy || hasActiveJob}
            loading={isActing && pendingAction === 'start'}
            size="lg"
            variant="outline"
            onClick={requestStart}
          >
            {t(tKey('startMove'))}
          </Button>
          <Button
            disabled={isBusy}
            loading={isActing && pendingAction === 'recover'}
            size="lg"
            variant="outline"
            onClick={requestRecover}
          >
            {t(tKey('recover'))}
          </Button>
        </HStack>
        {actionError ? (
          <Text color="fg.error" fontSize="md" role="alert">
            {actionError}
          </Text>
        ) : null}
      </Stack>
      <ConfirmDialog
        body={t(tKey(pendingAction === 'recover' ? 'recoverConfirmBody' : 'startConfirmBody'))}
        confirmLabel={t(tKey(pendingAction === 'recover' ? 'recover' : 'startMove'))}
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
        <Text color="fg.error" fontSize="md" role="alert">
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
    summary = t(tKey('statusNeedsRecovery'), { jobId: status.activeJobId });
  } else if (status.isRunning) {
    summary = t(tKey(status.operation === 'recovery' ? 'statusRecovering' : 'statusMoving'), {
      count: status.needsMoveCount,
    });
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
          {t(tKey('latestJob'), {
            jobId: status.latestJob.id,
            state: t(tKey(`jobStates.${status.latestJob.state}`)),
          })}
        </Text>
      ) : null}
      {errorMessage ? (
        <Text color="fg.error" fontSize="xs" whiteSpace="pre-wrap">
          {errorMessage}
        </Text>
      ) : null}
    </Stack>
  );
};
