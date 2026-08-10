import type { ModelInstallJob } from '@features/models/core/types';

import { Badge, Box, Collapsible, Flex, HStack, Icon, Spinner, Text } from '@chakra-ui/react';
import {
  cancelModelInstall,
  pauseModelInstall,
  pruneCompletedModelInstalls,
  resumeModelInstall,
} from '@features/models/data/api';
import {
  ensureInstallsLoaded,
  isActiveInstallStatus,
  refreshInstalls,
  useInstallsSelector,
} from '@features/models/data/installsStore';
import { setQueueExpanded, useModelsUiSelector } from '@features/models/ui/uiStore';
import { useNotify } from '@features/models/ui/useModelsNotify';
import { useMountEffect } from '@platform/react/useMountEffect';
import { getErrorMessage, useScopedAction } from '@platform/react/useScopedAction';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { Button, IconButton, Tooltip } from '@platform/ui';
import { ChevronUpIcon, ListOrderedIcon, PauseIcon, PlayIcon, RefreshCcwIcon, Trash2Icon, XIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { InstallQueueList } from './InstallQueueList';
import { getInstallJobDisplayName } from './queueUtils';

const TRIGGER_HOVER = { color: 'fg' } as const;
const INDICATOR_OPEN = { transform: 'rotate(180deg)' } as const;

/** Persistent, collapsible install queue footer for the model manager detail pane. */
export const InstallQueueBar = () => {
  const { t } = useTranslation();
  const notify = useNotify();
  const error = useInstallsSelector((snapshot) => snapshot.error);
  const jobs = useInstallsSelector((snapshot) => snapshot.jobs);
  const status = useInstallsSelector((snapshot) => snapshot.status);
  const queueExpanded = useModelsUiSelector((snapshot) => snapshot.queueExpanded);

  useMountEffect(() => {
    ensureInstallsLoaded();
  });

  const activeJobs = useMemo(() => jobs.filter((job) => isActiveInstallStatus(job.status)), [jobs]);
  const pausedJobs = useMemo(() => jobs.filter((job) => job.status === 'paused'), [jobs]);
  const pausableJobs = useMemo(() => jobs.filter((job) => job.status === 'downloading'), [jobs]);
  const cancellableJobs = useMemo(
    () => jobs.filter((job) => isActiveInstallStatus(job.status) || job.status === 'paused'),
    [jobs]
  );
  const settledCount = jobs.filter((job) => !isActiveInstallStatus(job.status) && job.status !== 'paused').length;

  // One busy flag per queue-wide control so each button tracks its own action.
  const { isBusy: isPausingAll, run: runPauseAll } = useScopedAction();
  const { isBusy: isResumingAll, run: runResumeAll } = useScopedAction();
  const { isBusy: isCancellingAll, run: runCancelAll } = useScopedAction();
  const { isBusy: isPruning, run: runPrune } = useScopedAction();
  const { isBusy: isRefreshing, run: runRefresh } = useScopedAction();

  const runBulk = useCallback(
    (
      run: ReturnType<typeof useScopedAction>['run'],
      call: (id: ModelInstallJob['id'], signal: AbortSignal) => Promise<unknown>,
      targets: ModelInstallJob[]
    ) =>
      run(
        async (owner) => {
          // allSettled: one failed job must not hide the others' outcomes,
          // and the requests that succeeded still deserve a fresh list.
          const results = await Promise.allSettled(targets.map((job) => call(job.id, owner.signal)));
          assertAccountScopeCurrent(owner);
          await refreshInstalls(owner);
          assertAccountScopeCurrent(owner);

          const failures = results.filter((result): result is PromiseRejectedResult => result.status === 'rejected');
          if (failures.length > 0) {
            notify.error(
              t('models.queueActionFailed'),
              t('models.queueActionPartialDescription', {
                error: getErrorMessage(failures[0]!.reason),
                failed: failures.length,
                total: targets.length,
              })
            );
          }
        },
        (message) => {
          notify.error(t('models.queueActionFailed'), message);
          void refreshInstalls();
        }
      ),
    [notify, t]
  );

  const handlePrune = useCallback(
    () =>
      runPrune(
        async (owner) => {
          await pruneCompletedModelInstalls(owner.signal);
          assertAccountScopeCurrent(owner);
          await refreshInstalls(owner);
          assertAccountScopeCurrent(owner);
        },
        (message) => notify.error(t('models.pruneFailed'), message)
      ),
    [notify, runPrune, t]
  );

  const handleRefresh = useCallback(() => runRefresh((owner) => refreshInstalls(owner)), [runRefresh]);
  const handleOpenChange = useCallback((event: { open: boolean }) => setQueueExpanded(event.open), []);
  const handlePauseAll = useCallback(
    () => void runBulk(runPauseAll, pauseModelInstall, pausableJobs),
    [pausableJobs, runBulk, runPauseAll]
  );
  const handleResumeAll = useCallback(
    () => void runBulk(runResumeAll, resumeModelInstall, pausedJobs),
    [pausedJobs, runBulk, runResumeAll]
  );
  const handleCancelAll = useCallback(
    () => void runBulk(runCancelAll, cancelModelInstall, cancellableJobs),
    [cancellableJobs, runBulk, runCancelAll]
  );

  const summary =
    activeJobs.length > 0
      ? `${getInstallJobDisplayName(activeJobs[0]!)}${activeJobs.length > 1 ? t('common.plusMore', { count: activeJobs.length - 1 }) : ''}`
      : jobs.length > 0
        ? t('models.installJobSummary', { count: jobs.length })
        : t('models.noInstallsYet');

  return (
    <Collapsible.Root
      bg="bg.subtle"
      borderTopWidth={1}
      flexShrink={0}
      open={queueExpanded}
      overflow="hidden"
      onOpenChange={handleOpenChange}
    >
      <Collapsible.Content>
        <Flex direction="column" h="min(22rem, 45dvh)" minH="0" overflow="hidden">
          <HStack borderBottomWidth={1} gap="2" justify="space-between" px="3" py="1.5">
            <Text color="fg.subtle" fontSize="2xs" fontWeight="700" textTransform="uppercase">
              {t('models.installQueue')}
            </Text>
            <HStack gap="1">
              <Button loading={isRefreshing} size="2xs" variant="ghost" onClick={handleRefresh}>
                <Icon as={RefreshCcwIcon} boxSize="3" />
                {t('common.refresh')}
              </Button>
              <Button
                disabled={settledCount === 0}
                loading={isPruning}
                size="2xs"
                variant="ghost"
                onClick={handlePrune}
              >
                <Icon as={Trash2Icon} boxSize="3" />
                {t('models.clearFinished')}
              </Button>
            </HStack>
          </HStack>
          <Box flex="1" minH="0" overflow="hidden" p="2">
            <InstallQueueList error={error} jobs={jobs} status={status} />
          </Box>
        </Flex>
      </Collapsible.Content>

      <HStack gap="1" px="3" py="2">
        <Collapsible.Trigger
          alignItems="center"
          bg="transparent"
          color="inherit"
          display="flex"
          flex="1"
          gap="2"
          minW="0"
          textAlign="start"
          _hover={TRIGGER_HOVER}
        >
          {activeJobs.length > 0 ? (
            <Spinner borderWidth="1.5px" boxSize="3.5" color="accent.solid" flexShrink={0} />
          ) : (
            <Icon as={ListOrderedIcon} boxSize="3.5" color="fg.subtle" flexShrink={0} />
          )}
          <Text flex="1" fontSize="xs" fontWeight="600" minW="0" truncate>
            {summary}
          </Text>

          {activeJobs.length > 0 ? (
            <Badge colorPalette="accent" flexShrink={0} fontSize="2xs" size="sm" variant="solid">
              {activeJobs.length}
            </Badge>
          ) : null}
          <Collapsible.Indicator _open={INDICATOR_OPEN} transition="transform var(--wb-motion-duration-slow)">
            <Icon as={ChevronUpIcon} boxSize="4" color="fg.subtle" flexShrink={0} />
          </Collapsible.Indicator>
        </Collapsible.Trigger>

        <HStack flexShrink={0} gap="0.5">
          {pausableJobs.length > 0 ? (
            <Tooltip content={t('models.pauseAllDownloads')}>
              <IconButton
                aria-label={t('models.pauseAllInstalls')}
                loading={isPausingAll}
                size="2xs"
                variant="ghost"
                onClick={handlePauseAll}
              >
                <Icon as={PauseIcon} boxSize="3.5" />
              </IconButton>
            </Tooltip>
          ) : null}
          {pausedJobs.length > 0 ? (
            <Tooltip content={t('models.resumeAllPausedDownloads')}>
              <IconButton
                aria-label={t('models.resumeAllInstalls')}
                loading={isResumingAll}
                size="2xs"
                variant="ghost"
                onClick={handleResumeAll}
              >
                <Icon as={PlayIcon} boxSize="3.5" />
              </IconButton>
            </Tooltip>
          ) : null}
          {cancellableJobs.length > 0 ? (
            <Tooltip content={t('models.cancelAllInstalls')}>
              <IconButton
                aria-label={t('models.cancelAllInstalls')}
                loading={isCancellingAll}
                size="2xs"
                variant="ghost"
                onClick={handleCancelAll}
              >
                <Icon as={XIcon} boxSize="3.5" />
              </IconButton>
            </Tooltip>
          ) : null}
        </HStack>
      </HStack>
    </Collapsible.Root>
  );
};
