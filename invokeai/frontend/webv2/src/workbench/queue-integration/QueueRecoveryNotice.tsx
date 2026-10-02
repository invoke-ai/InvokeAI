import { Alert, Button, HStack, Spinner, Stack, Text } from '@chakra-ui/react';
import { downloadText } from '@platform/browser/downloadBlob';
import { useMountEffect } from '@platform/react/useMountEffect';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { useQueryClient } from '@tanstack/react-query';
import { useCallback, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import {
  createQueueRecoveryExport,
  discardQueueRecoveryProject,
  getQueueRecoveryExportName,
  QueueRecoveryError,
} from './queueRecovery';
import { getQueueRecoveryProjectIdsQueryKey, useQueueRecoveryProjectIds } from './useQueueRecoveryProjectIds';

type PendingAction = 'discard' | 'export' | 'open';

const getErrorKey = (error: unknown): string => {
  if (error instanceof QueueRecoveryError) {
    if (error.code === 'lock-contended') {
      return 'shell.queueRecovery.lockContended';
    }
    if (error.code === 'lock-unavailable') {
      return 'shell.queueRecovery.lockUnavailable';
    }
    return 'shell.queueRecovery.storageUnavailable';
  }
  return 'shell.queueRecovery.actionFailed';
};

export const QueueRecoveryNotice = ({
  onOpen,
  openProjectIds,
}: {
  onOpen?: (projectId: string) => Promise<void>;
  openProjectIds?: readonly string[];
}) => {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [owner] = useState(captureAccountScope);
  const query = useQueueRecoveryProjectIds(owner);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [pendingAction, setPendingAction] = useState<PendingAction | null>(null);
  const [discardProjectId, setDiscardProjectId] = useState<string | null>(null);
  const [errorKey, setErrorKey] = useState<string | null>(null);
  const isMountedRef = useRef(true);
  const actionInFlightRef = useRef(false);

  useMountEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  });

  const openIds = new Set(openProjectIds);
  const projectIds =
    query.data?.kind === 'available' ? query.data.projectIds.filter((projectId) => !openIds.has(projectId)) : [];
  const visibleIndex = Math.min(selectedIndex, Math.max(0, projectIds.length - 1));
  const projectId = projectIds[visibleIndex];
  const isBusy = pendingAction !== null || discardProjectId !== null;

  const canUpdate = useCallback(() => isMountedRef.current && isAccountScopeCurrent(owner), [owner]);
  const exportProject = useCallback(async () => {
    if (!projectId || isBusy || actionInFlightRef.current || !canUpdate()) {
      return;
    }
    const selectedProjectId = projectId;
    actionInFlightRef.current = true;
    setErrorKey(null);
    setPendingAction('export');
    try {
      const json = await createQueueRecoveryExport(owner, selectedProjectId);
      if (!canUpdate()) {
        return;
      }
      downloadText(json, getQueueRecoveryExportName(selectedProjectId), 'application/json');
      await queryClient.invalidateQueries({ queryKey: getQueueRecoveryProjectIdsQueryKey(owner) });
    } catch (error) {
      if (canUpdate()) {
        setErrorKey(getErrorKey(error));
      }
    } finally {
      actionInFlightRef.current = false;
      if (canUpdate()) {
        setPendingAction(null);
      }
    }
  }, [canUpdate, isBusy, owner, projectId, queryClient]);
  const openProject = useCallback(async () => {
    if (!onOpen || !projectId || isBusy || actionInFlightRef.current || !canUpdate()) {
      return;
    }
    const selectedProjectId = projectId;
    actionInFlightRef.current = true;
    setErrorKey(null);
    setPendingAction('open');
    try {
      await onOpen(selectedProjectId);
    } catch {
      if (canUpdate()) {
        setErrorKey('shell.queueRecovery.openFailed');
      }
    } finally {
      actionInFlightRef.current = false;
      if (canUpdate()) {
        setPendingAction(null);
      }
    }
  }, [canUpdate, isBusy, onOpen, projectId]);
  const confirmDiscard = useCallback(async () => {
    const selectedProjectId = discardProjectId;
    if (!selectedProjectId || pendingAction !== null || actionInFlightRef.current) {
      return;
    }
    actionInFlightRef.current = true;
    setErrorKey(null);
    setPendingAction('discard');
    try {
      await discardQueueRecoveryProject(owner, selectedProjectId);
      if (!canUpdate()) {
        return;
      }
      await queryClient.invalidateQueries({ queryKey: getQueueRecoveryProjectIdsQueryKey(owner) });
    } catch (error) {
      if (canUpdate()) {
        setErrorKey(getErrorKey(error));
      }
    } finally {
      actionInFlightRef.current = false;
      if (canUpdate()) {
        setPendingAction(null);
        setDiscardProjectId(null);
      }
    }
  }, [canUpdate, discardProjectId, owner, pendingAction, queryClient]);
  const selectPrevious = useCallback(() => {
    setErrorKey(null);
    setSelectedIndex((index) => Math.max(0, index - 1));
  }, []);
  const selectNext = useCallback(() => {
    setErrorKey(null);
    setSelectedIndex((index) => Math.min(projectIds.length - 1, index + 1));
  }, [projectIds.length]);
  const openDiscardConfirmation = useCallback(() => {
    if (projectId && !isBusy && canUpdate()) {
      setDiscardProjectId(projectId);
    }
  }, [canUpdate, isBusy, projectId]);
  const closeDiscardConfirmation = useCallback(() => {
    if (pendingAction === null) {
      setDiscardProjectId(null);
    }
  }, [pendingAction]);
  const handleExport = useCallback(() => void exportProject(), [exportProject]);
  const handleOpen = useCallback(() => void openProject(), [openProject]);

  if (query.isPending) {
    return (
      <HStack color="fg.muted" px="4" py="2">
        <Spinner size="xs" />
        <Text fontSize="sm">{t('shell.queueRecovery.loading')}</Text>
      </HStack>
    );
  }
  if (query.data?.kind === 'unavailable' || query.isError) {
    return (
      <Alert.Root borderRadius="none" status="warning" variant="surface">
        <Alert.Indicator />
        <Alert.Content>
          <Alert.Title>{t('shell.queueRecovery.storageUnavailableTitle')}</Alert.Title>
          <Alert.Description>{t('shell.queueRecovery.storageUnavailable')}</Alert.Description>
        </Alert.Content>
      </Alert.Root>
    );
  }
  if (!projectId) {
    return null;
  }

  return (
    <>
      <Alert.Root borderRadius="none" status="warning" variant="surface">
        <Alert.Indicator />
        <Alert.Content>
          <HStack align="center" flexWrap="wrap" gap="4" justify="space-between" w="full">
            <Stack gap="0.5" minW="0">
              <Alert.Title>{t('shell.queueRecovery.title')}</Alert.Title>
              <Alert.Description overflowWrap="anywhere">
                {t('shell.queueRecovery.description', { projectId })}
                {projectIds.length > 1
                  ? ` ${t('shell.queueRecovery.position', {
                      count: projectIds.length,
                      number: visibleIndex + 1,
                    })}`
                  : null}
              </Alert.Description>
              {errorKey ? <Text color="fg.error">{t(errorKey)}</Text> : null}
            </Stack>
            <HStack flexShrink="0" flexWrap="wrap">
              {projectIds.length > 1 ? (
                <>
                  <Button disabled={isBusy || visibleIndex === 0} onClick={selectPrevious} size="sm" variant="ghost">
                    {t('shell.queueRecovery.previous')}
                  </Button>
                  <Button
                    disabled={isBusy || visibleIndex === projectIds.length - 1}
                    onClick={selectNext}
                    size="sm"
                    variant="ghost"
                  >
                    {t('shell.queueRecovery.next')}
                  </Button>
                </>
              ) : null}
              {onOpen ? (
                <Button disabled={isBusy} loading={pendingAction === 'open'} onClick={handleOpen} size="sm">
                  {t('shell.queueRecovery.open')}
                </Button>
              ) : null}
              <Button
                disabled={isBusy}
                loading={pendingAction === 'export'}
                onClick={handleExport}
                size="sm"
                variant="outline"
              >
                {t('shell.queueRecovery.export')}
              </Button>
              <Button disabled={isBusy} onClick={openDiscardConfirmation} size="sm" variant="ghost">
                {t('shell.queueRecovery.discard')}
              </Button>
            </HStack>
          </HStack>
        </Alert.Content>
      </Alert.Root>
      <ConfirmDialog
        body={t('shell.queueRecovery.discardConfirmBody')}
        confirmLabel={t('shell.queueRecovery.discard')}
        isDestructive
        isOpen={discardProjectId !== null}
        title={t('shell.queueRecovery.discardConfirmTitle')}
        onClose={closeDiscardConfirmation}
        onConfirm={confirmDiscard}
      />
    </>
  );
};
