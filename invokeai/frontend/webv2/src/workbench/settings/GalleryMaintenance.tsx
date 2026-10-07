import type { components } from '@api/schema';
import type { AccountScope } from '@platform/state/accountLifecycle';

import { Box, Flex, Stack, Text } from '@chakra-ui/react';
import { invalidateGallery } from '@features/gallery/queries';
import { useMountEffect } from '@platform/react/useMountEffect';
import {
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { apiFetchJson, getApiErrorMessage, ApiError } from '@platform/transport/http';
import { Button, ConfirmDialog } from '@platform/ui';
import { useQueryClient } from '@tanstack/react-query';
import { refreshImageMapPoints } from '@workbench/image-map/imageMapStore';
import { createElement, useCallback, useId, useMemo, useRef, useState, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

type GalleryMaintenanceOperation = components['schemas']['GalleryMaintenanceOperation'];
type GalleryMaintenancePreview = components['schemas']['GalleryMaintenancePreview'];
type GalleryMaintenanceResult = components['schemas']['GalleryMaintenanceResult'];

const ACTIONS: Record<
  GalleryMaintenanceOperation,
  { label: string; confirmLabel: string; description: string; title: string }
> = {
  remove_missing: {
    confirmLabel: 'removeMissingConfirm',
    description: 'removeMissingDescription',
    label: 'removeMissing',
    title: 'removeMissingTitle',
  },
  archive_untracked: {
    confirmLabel: 'archiveUntrackedConfirm',
    description: 'archiveUntrackedDescription',
    label: 'archiveUntracked',
    title: 'archiveUntrackedTitle',
  },
  regenerate_thumbnails: {
    confirmLabel: 'regenerateThumbnailsConfirm',
    description: 'regenerateThumbnailsDescription',
    label: 'regenerateThumbnails',
    title: 'regenerateThumbnailsTitle',
  },
};

type PreviewState =
  | { kind: 'idle' }
  | { kind: 'loading'; owner: AccountScope }
  | { kind: 'ready'; owner: AccountScope; value: GalleryMaintenancePreview }
  | { kind: 'error'; owner: AccountScope; message: string };

type LatestOutcome =
  | { kind: 'result'; owner: AccountScope; value: GalleryMaintenanceResult; refreshError?: string }
  | {
      kind: 'error';
      owner: AccountScope;
      operation: GalleryMaintenanceOperation;
      message: string;
      isConflict: boolean;
    };

const routeForOperation = (operation: GalleryMaintenanceOperation): string =>
  `/api/v1/app/gallery/maintenance/${
    operation === 'remove_missing'
      ? 'remove-missing'
      : operation === 'archive_untracked'
        ? 'archive-untracked'
        : 'regenerate-thumbnails'
  }`;

const tKey = (key: string): string => `settings.galleryMaintenance.${key}`;
const ROW_ALIGN = { base: 'stretch', md: 'center' } as const;
const ROW_DIRECTION = { base: 'column', md: 'row' } as const;

export const GalleryMaintenance = () => {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const resourceId = useId();
  const requestSequence = useRef(0);
  const executing = useRef(false);
  const isMounted = useRef(true);
  const [operation, setOperation] = useState<GalleryMaintenanceOperation | null>(null);
  const [preview, setPreview] = useState<PreviewState>({ kind: 'idle' });
  const [isExecuting, setIsExecuting] = useState(false);
  const [latest, setLatest] = useState<LatestOutcome | null>(null);

  useMountEffect(() => {
    isMounted.current = true;
    const unregister = registerAccountOwnedResource({
      clear: () => {
        requestSequence.current += 1;
        setOperation(null);
        setPreview({ kind: 'idle' });
        setLatest(null);
      },
      name: `gallery-maintenance:${resourceId}`,
    });
    return () => {
      isMounted.current = false;
      requestSequence.current += 1;
      unregister();
    };
  });

  const startPreview = useCallback(
    async (nextOperation: GalleryMaintenanceOperation) => {
      setOperation(nextOperation);
      const owner = captureAccountScope();
      const sequence = ++requestSequence.current;
      setPreview({ kind: 'loading', owner });

      try {
        const value = await apiFetchJson<GalleryMaintenancePreview>('/api/v1/app/gallery/maintenance/preview', {
          method: 'POST',
          body: JSON.stringify({ operation: nextOperation }),
          signal: owner.signal,
        });
        if (sequence === requestSequence.current && isAccountScopeCurrent(owner)) {
          setPreview({ kind: 'ready', owner, value });
        }
      } catch (error) {
        if (sequence === requestSequence.current && isAccountScopeCurrent(owner)) {
          setPreview({
            kind: 'error',
            owner,
            message: getApiErrorMessage(error, t(tKey('previewFailed'))),
          });
        }
      }
    },
    [t]
  );

  const closeDialog = useCallback(() => {
    requestSequence.current += 1;
    setOperation(null);
    setPreview({ kind: 'idle' });
  }, []);

  const retryPreview = useCallback(() => {
    if (operation !== null) {
      void startPreview(operation);
    }
  }, [operation, startPreview]);

  const previewHandlers = useMemo<Record<GalleryMaintenanceOperation, () => void>>(
    () => ({
      remove_missing: () => void startPreview('remove_missing'),
      archive_untracked: () => void startPreview('archive_untracked'),
      regenerate_thumbnails: () => void startPreview('regenerate_thumbnails'),
    }),
    [startPreview]
  );

  const execute = useCallback(async () => {
    if (executing.current || operation === null || preview.kind !== 'ready' || preview.value.operation !== operation) {
      return;
    }
    const owner = preview.owner;
    if (!isAccountScopeCurrent(owner)) {
      return;
    }

    executing.current = true;
    setIsExecuting(true);
    try {
      const value = await apiFetchJson<GalleryMaintenanceResult>(routeForOperation(operation), {
        method: 'POST',
        body: JSON.stringify({ fingerprint: preview.value.fingerprint }),
        signal: owner.signal,
      });
      if (!isAccountScopeCurrent(owner)) {
        return;
      }

      let refreshError: string | undefined;
      if (operation === 'remove_missing' && value.records_removed > 0) {
        const refreshErrors: string[] = [];
        try {
          await invalidateGallery(queryClient, owner);
        } catch (error) {
          refreshErrors.push(getApiErrorMessage(error, t(tKey('refreshFailed'))));
        }
        if (!isAccountScopeCurrent(owner)) {
          return;
        }
        try {
          await refreshImageMapPoints();
        } catch (error) {
          refreshErrors.push(getApiErrorMessage(error, t(tKey('refreshFailed'))));
        }
        if (!isAccountScopeCurrent(owner)) {
          return;
        }
        if (refreshErrors.length > 0) {
          refreshError = refreshErrors.join('\n');
        }
      }
      if (isMounted.current && isAccountScopeCurrent(owner)) {
        setLatest({ kind: 'result', owner, value, refreshError });
      }
    } catch (error) {
      if (isMounted.current && isAccountScopeCurrent(owner)) {
        setLatest({
          kind: 'error',
          owner,
          operation,
          message: getApiErrorMessage(error, t(tKey('executeFailed'))),
          isConflict: error instanceof ApiError && error.status === 409,
        });
      }
    } finally {
      executing.current = false;
      if (isMounted.current) {
        setIsExecuting(false);
      }
    }
  }, [operation, preview, queryClient, t]);

  const isCurrentPreview =
    preview.kind === 'ready' &&
    operation !== null &&
    preview.value.operation === operation &&
    isAccountScopeCurrent(preview.owner);
  const isPreviewLoading = preview.kind === 'loading' && isAccountScopeCurrent(preview.owner);
  const isBusy = isExecuting || isPreviewLoading;
  const visibleLatest = latest && isAccountScopeCurrent(latest.owner) ? latest : null;

  const renderPreview = useCallback(() => {
    if (preview.kind === 'idle' || operation === null) {
      return null;
    }
    const owner = preview.owner;
    if (!isAccountScopeCurrent(owner)) {
      return (
        <Stack gap="2">
          <Text>{t(tKey('accountChanged'))}</Text>
          <Button alignSelf="start" variant="outline" onClick={retryPreview}>
            {t(tKey('retryPreview'))}
          </Button>
        </Stack>
      );
    }
    if (preview.kind === 'loading') {
      return (
        <Text aria-live="polite" role="status">
          {t(tKey('previewLoading'))}
        </Text>
      );
    }
    if (preview.kind === 'error') {
      return (
        <Stack gap="2">
          <Text color="fg.error" role="alert">
            {preview.message}
          </Text>
          <Button alignSelf="start" variant="outline" onClick={retryPreview}>
            {t(tKey('retryPreview'))}
          </Button>
        </Stack>
      );
    }

    return (
      <Stack gap="2">
        <Text>
          {t(tKey('previewSummary'), {
            affected: preview.value.affected_count,
            examined: preview.value.examined_count,
            errors: preview.value.error_count,
            skipped: preview.value.skipped_count,
          })}
        </Text>
        {preview.value.archive_path ? (
          <Text overflowWrap="anywhere">
            {t(tKey('archiveLocation'))}: {preview.value.archive_path}
          </Text>
        ) : null}
        {(preview.value.errors?.length ?? 0) > 0 ? (
          <Text color="fg.warning" role="status" whiteSpace="pre-wrap">
            {preview.value.errors?.join('\n')}
          </Text>
        ) : null}
      </Stack>
    );
  }, [operation, preview, retryPreview, t]);

  const confirmationBody = useMemo(
    () => createElement(GalleryMaintenanceConfirmBody, { isExecuting, operation, renderPreview }),
    [isExecuting, operation, renderPreview]
  );

  return (
    <Stack gap="0">
      <Text as="h3" fontSize="lg" fontWeight="600" pb="1">
        {t(tKey('heading'))}
      </Text>
      {Object.entries(ACTIONS).map(([operationId, action]) => {
        const actionOperation = operationId as GalleryMaintenanceOperation;
        return (
          <Box key={actionOperation} py="4" borderBottomWidth="1px" borderColor="border.subtle">
            <Flex align={ROW_ALIGN} direction={ROW_DIRECTION} gap="3">
              <Stack flex="1" gap="1">
                <Text fontSize="lg" fontWeight="500">
                  {t(tKey(action.label))}
                </Text>
                <Text color="fg.muted" fontSize="md">
                  {t(tKey(action.description))}
                </Text>
              </Stack>
              <Button
                disabled={isBusy}
                flexShrink={0}
                size="lg"
                variant="outline"
                onClick={previewHandlers[actionOperation]}
              >
                {t(tKey(action.label))}
              </Button>
            </Flex>
          </Box>
        );
      })}
      {visibleLatest ? <LatestOutcomeView outcome={visibleLatest} retryPreview={startPreview} /> : null}
      {isExecuting && operation === null ? (
        <Text aria-live="polite" color="fg.muted" role="status" py="3">
          {t(tKey('executionRunning'))}
        </Text>
      ) : null}
      <ConfirmDialog
        body={confirmationBody}
        confirmLabel={operation ? t(tKey(ACTIONS[operation].confirmLabel)) : t(tKey('proceed'))}
        isConfirmDisabled={!isCurrentPreview}
        isDestructive
        isOpen={operation !== null}
        title={operation ? t(tKey(ACTIONS[operation].title)) : ''}
        onClose={closeDialog}
        onConfirm={execute}
      />
    </Stack>
  );
};

const GalleryMaintenanceConfirmBody = ({
  isExecuting,
  operation,
  renderPreview,
}: {
  isExecuting: boolean;
  operation: GalleryMaintenanceOperation | null;
  renderPreview: () => ReactNode;
}) => {
  const { t } = useTranslation();

  return (
    <Stack gap="3">
      <Text>{operation ? t(tKey(ACTIONS[operation].description)) : ''}</Text>
      <Text color="fg.muted">{t(tKey('installationScope'))}</Text>
      {operation === 'remove_missing' ? <Text color="fg.muted">{t(tKey('removeMissingConsequence'))}</Text> : null}
      {isExecuting ? (
        <Text aria-live="polite" role="status">
          {t(tKey('executionRunning'))}
        </Text>
      ) : (
        renderPreview()
      )}
    </Stack>
  );
};

const LatestOutcomeView = ({
  outcome,
  retryPreview,
}: {
  outcome: LatestOutcome;
  retryPreview: (operation: GalleryMaintenanceOperation) => void;
}) => {
  const { t } = useTranslation();
  const retryOperation = outcome.kind === 'error' ? outcome.operation : null;
  const handleRetryPreview = useCallback(() => {
    if (retryOperation !== null) {
      retryPreview(retryOperation);
    }
  }, [retryOperation, retryPreview]);

  if (outcome.kind === 'error') {
    return (
      <Stack data-testid="gallery-maintenance-result" gap="2" py="4" role="status">
        <Text fontWeight="600">{t(tKey(outcome.isConflict ? 'conflict' : 'operationFailed'))}</Text>
        <Text color="fg.error">{outcome.message}</Text>
        {outcome.isConflict ? (
          <Button alignSelf="start" variant="outline" onClick={handleRetryPreview}>
            {t(tKey('reviewCurrentItems'))}
          </Button>
        ) : null}
      </Stack>
    );
  }

  const { value } = outcome;
  return (
    <Stack data-testid="gallery-maintenance-result" gap="2" py="4" role="status">
      <Text fontWeight="600">{t(tKey(value.status))}</Text>
      <Text>
        {t(tKey('resultSummary'), {
          examined: value.examined_count,
          failed: value.failed_count,
          imagesArchived: value.images_archived,
          recordsRemoved: value.records_removed,
          skipped: value.skipped_count,
          thumbnailsArchived: value.thumbnails_archived,
          thumbnailsRegenerated: value.thumbnails_regenerated,
        })}
      </Text>
      {value.archive_path ? (
        <Text overflowWrap="anywhere">
          {t(tKey('archiveLocation'))}: {value.archive_path}
        </Text>
      ) : null}
      {value.backup_path ? (
        <Text overflowWrap="anywhere">
          {t(tKey('backupLocation'))}: {value.backup_path}
        </Text>
      ) : null}
      {(value.errors?.length ?? 0) > 0 ? (
        <Text color="fg.warning" whiteSpace="pre-wrap">
          {value.errors?.join('\n')}
        </Text>
      ) : null}
      {outcome.refreshError ? <Text color="fg.warning">{outcome.refreshError}</Text> : null}
    </Stack>
  );
};
