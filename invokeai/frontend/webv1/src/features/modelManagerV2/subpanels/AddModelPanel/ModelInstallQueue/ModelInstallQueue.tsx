import type { SystemStyleObject } from '@invoke-ai/ui-library';
import {
  Box,
  Button,
  ButtonGroup,
  Flex,
  Heading,
  IconButton,
  Menu,
  MenuButton,
  MenuItem,
  MenuList,
  Table,
  Tbody,
  Td,
  Text,
  Th,
  Thead,
  Tr,
} from '@invoke-ai/ui-library';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { getApiErrorDetail } from 'features/modelManagerV2/util/getApiErrorDetail';
import { toast } from 'features/toast/toast';
import { memo, useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiBroomBold, PiCaretDownBold, PiPauseFill, PiPlayFill, PiXBold } from 'react-icons/pi';
import {
  useCancelModelInstallMutation,
  useListModelInstallsQuery,
  usePauseModelInstallMutation,
  usePruneCompletedModelInstallsMutation,
  useResumeModelInstallMutation,
} from 'services/api/endpoints/models';
import type { ModelInstallJob } from 'services/api/types';

import { ModelInstallQueueItem } from './ModelInstallQueueItem';

const hasRestartRequired = (job: ModelInstallJob) => {
  return job.download_parts?.some((part) => part.resume_required || part.status === 'error') ?? false;
};

const ModelQueueTableSx: SystemStyleObject = {
  '& tbody tr:nth-of-type(odd)': {
    backgroundColor: 'rgba(255, 255, 255, 0.04)',
  },
  '& tbody tr:nth-of-type(even)': {
    backgroundColor: 'transparent',
  },
  'td, th': {
    borderColor: 'base.700',
  },

  th: {
    position: 'sticky',
    top: 0,
    zIndex: 1,
    backgroundColor: 'base.800',
    py: 2,
  },

  'th:first-of-type': {
    borderTopLeftRadius: 'base',
  },
  'th:last-of-type': {
    borderTopRightRadius: 'base',
  },
  'tr:last-of-type td:first-of-type': {
    borderBottomLeftRadius: 'base',
  },
  'tr:last-of-type td:last-of-type': {
    borderBottomRightRadius: 'base',
  },
};

export const ModelInstallQueue = memo(() => {
  const { t } = useTranslation();
  const { data } = useListModelInstallsQuery();
  const [bulkActionInProgress, setBulkActionInProgress] = useState<'pause' | 'resume' | 'cancel' | null>(null);
  const bulkActionLockRef = useRef(false);

  const reversedData = useMemo(() => {
    return data?.toReversed() ?? [];
  }, [data]);

  const [cancelModelInstall] = useCancelModelInstallMutation();
  const [pauseModelInstall] = usePauseModelInstallMutation();
  const [resumeModelInstall] = useResumeModelInstallMutation();
  const [_pruneCompletedModelInstalls, { isLoading: isPruning }] = usePruneCompletedModelInstallsMutation();

  const installActionIds = useMemo(() => {
    const pauseable: number[] = [];
    const resumable: number[] = [];
    const cancelable: number[] = [];

    for (const model of data ?? []) {
      if (model.status === 'downloading' || model.status === 'waiting') {
        pauseable.push(model.id);
      }

      if (model.status === 'paused') {
        cancelable.push(model.id);
        if (!hasRestartRequired(model)) {
          resumable.push(model.id);
        }
        continue;
      }

      if (model.status === 'running' || model.status === 'downloads_done') {
        cancelable.push(model.id);
      }
    }

    return {
      pauseable,
      resumable,
      cancelable: cancelable.concat(pauseable),
    } as const;
  }, [data]);

  const {
    pauseable: pauseableInstallIds,
    resumable: resumableInstallIds,
    cancelable: cancelableInstallIds,
  } = installActionIds;

  const runBulkAction = useCallback(
    async (action: 'pause' | 'resume' | 'cancel', installIds: number[]) => {
      if (installIds.length === 0 || bulkActionLockRef.current || isPruning) {
        return;
      }

      bulkActionLockRef.current = true;
      setBulkActionInProgress(action);

      try {
        const results = await Promise.allSettled(
          installIds.map((id) => {
            if (action === 'pause') {
              return pauseModelInstall(id).unwrap();
            }
            if (action === 'resume') {
              return resumeModelInstall(id).unwrap();
            }
            return cancelModelInstall(id).unwrap();
          })
        );

        const hasSucceeded = results.some((result) => result.status === 'fulfilled');
        const firstError = results.find((result) => result.status === 'rejected') as PromiseRejectedResult | undefined;

        if (hasSucceeded) {
          const title =
            action === 'pause'
              ? t('toast.modelDownloadPaused')
              : action === 'resume'
                ? t('toast.modelDownloadResumed')
                : t('toast.modelImportCanceled');

          toast({
            id: `MODEL_INSTALL_QUEUE_${action.toUpperCase()}_ALL`,
            title,
            status: 'success',
          });
        }

        if (firstError) {
          toast({
            id: `MODEL_INSTALL_QUEUE_${action.toUpperCase()}_ALL_FAILED`,
            title: getApiErrorDetail(firstError.reason),
            status: 'error',
          });
        }
      } finally {
        bulkActionLockRef.current = false;
        setBulkActionInProgress(null);
      }
    },
    [cancelModelInstall, isPruning, pauseModelInstall, resumeModelInstall, t]
  );

  const pruneCompletedModelInstalls = useCallback(async () => {
    if (bulkActionLockRef.current) {
      return;
    }

    try {
      await _pruneCompletedModelInstalls().unwrap();
      toast({
        id: 'MODEL_INSTALL_QUEUE_PRUNED',
        title: t('toast.prunedQueue'),
        status: 'success',
      });
    } catch (error) {
      toast({
        id: 'MODEL_INSTALL_QUEUE_PRUNE_FAILED',
        title: getApiErrorDetail(error),
        status: 'error',
      });
    }
  }, [_pruneCompletedModelInstalls, t]);

  const hasPauseableInstalls = pauseableInstallIds.length > 0;
  const hasResumableInstalls = resumableInstallIds.length > 0;
  const hasCancelableInstalls = cancelableInstallIds.length > 0;

  const pruneAvailable = useMemo(() => {
    return data?.some(
      (model) => model.status === 'cancelled' || model.status === 'error' || model.status === 'completed'
    );
  }, [data]);

  const pauseAll = useCallback(() => {
    void runBulkAction('pause', pauseableInstallIds);
  }, [pauseableInstallIds, runBulkAction]);

  const resumeAll = useCallback(() => {
    void runBulkAction('resume', resumableInstallIds);
  }, [resumableInstallIds, runBulkAction]);

  const cancelAll = useCallback(() => {
    void runBulkAction('cancel', cancelableInstallIds);
  }, [cancelableInstallIds, runBulkAction]);

  const isBulkActionRunning = bulkActionInProgress !== null;

  return (
    <Flex flexDir="column" h="full" gap={4}>
      {/* Model Queue Header */}
      <Flex justifyContent="space-between" alignItems="center">
        <Flex alignItems="center" gap={2}>
          <Heading size="md">{t('modelManager.installQueue')}</Heading>
        </Flex>

        {/* Bulk Actions */}
        {/* Non-destructive, easily-ccessible actions */}
        <Flex gap={2}>
          {hasPauseableInstalls && (
            <Button
              size="sm"
              leftIcon={<PiPauseFill />}
              isDisabled={isBulkActionRunning || isPruning}
              onClick={pauseAll}
              variant="outline"
            >
              {t('modelManager.pauseAll')}
            </Button>
          )}

          {hasResumableInstalls && (
            <Button
              size="sm"
              leftIcon={<PiPlayFill />}
              isDisabled={isBulkActionRunning || isPruning}
              onClick={resumeAll}
              variant="outline"
            >
              {t('modelManager.resumeAll')}
            </Button>
          )}

          {/* Destructive Actions go to the button group/menu */}
          <ButtonGroup>
            <Button
              leftIcon={<PiBroomBold />}
              size="sm"
              isDisabled={!pruneAvailable || isBulkActionRunning || isPruning}
              onClick={pruneCompletedModelInstalls}
              variant="outline"
            >
              {t('modelManager.prune')}
            </Button>
            <Menu>
              <MenuButton
                as={IconButton}
                size="sm"
                aria-label={t('accessibility.menu')}
                icon={<PiCaretDownBold />}
                disabled={!pruneAvailable && !hasCancelableInstalls}
              />
              <MenuList>
                <MenuItem
                  color="error.300"
                  icon={<PiXBold />}
                  isDisabled={!hasCancelableInstalls || isBulkActionRunning || isPruning}
                  onClick={cancelAll}
                  isDestructive
                >
                  {t('modelManager.cancelAll')}
                </MenuItem>
              </MenuList>
            </Menu>
          </ButtonGroup>
        </Flex>
      </Flex>

      {/* Model Queue List */}
      <Box layerStyle="second" borderRadius="base" w="full" h="full">
        <ScrollableContent>
          <Table size="sm" sx={ModelQueueTableSx}>
            <Thead>
              <Tr>
                <Th minWidth="50px"></Th>
                <Th width="80%">Name</Th>
                <Th minWidth="130px">Status</Th>
                <Th minWidth="160px" textAlign="right">
                  Actions
                </Th>
              </Tr>
            </Thead>
            <Tbody>
              {data?.length === 0 ? (
                <Tr>
                  <Td colSpan={4} textAlign="center" py={8}>
                    <Text variant="subtext">{t('modelManager.queueEmpty')}</Text>
                  </Td>
                </Tr>
              ) : (
                reversedData?.map((model) => <ModelInstallQueueItem key={model.id} installJob={model} />)
              )}
            </Tbody>
          </Table>
        </ScrollableContent>
      </Box>
    </Flex>
  );
});

ModelInstallQueue.displayName = 'ModelInstallQueue';
