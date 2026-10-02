import type { ModelInstallJob } from '@features/models/core/types';

import { Badge, Box, Flex, HStack, Icon, Menu, Portal, Separator, Spinner, Text } from '@chakra-ui/react';
import {
  cancelModelInstall,
  pauseModelInstall,
  pruneCompletedModelInstalls,
  resumeModelInstall,
} from '@features/models/data/api';
import {
  ensureInstallsLoaded,
  refreshInstalls,
  useInstallProgress,
  useInstallsSelector,
} from '@features/models/data/installsStore';
import { setQueueExpanded, setQueueMaximized, useModelsUiSelector } from '@features/models/ui/uiStore';
import { useNotify } from '@features/models/ui/useModelsNotify';
import { useMountEffect } from '@platform/react/useMountEffect';
import { getErrorMessage, useScopedAction } from '@platform/react/useScopedAction';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { Button, Group, IconButton, Tooltip, useTooltipTriggerIds } from '@platform/ui';
import { MenuActionItem, MenuContent } from '@platform/ui/Menu';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import {
  ChevronDownIcon,
  ChevronUpIcon,
  Maximize2Icon,
  Minimize2Icon,
  PauseIcon,
  PlayIcon,
  RefreshCcwIcon,
  Trash2Icon,
  TriangleAlertIcon,
  XIcon,
} from 'lucide-react';
import { useCallback, useId, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { InstallQueueRow, InstallQueueSummary } from './queueModel';

import { InstallProgressBar, useInstallProgressCaption } from './InstallProgressCell';
import { InstallQueueTable } from './InstallQueueTable';
import {
  buildInstallQueueRows,
  countFinishedInstallJobs,
  getInstallJobDisplayName,
  isActiveRowStatus,
  resolveInstallProgress,
  summarizeInstallQueue,
} from './queueModel';
import { useInstallJobActions } from './useInstallJobActions';

const DOCKED_HEIGHT = 'min(30rem, 60dvh)';
const TOGGLE_HOVER = { color: 'fg' } as const;
const BULK_MENU_POSITIONING = { placement: 'bottom-start' } as const;

/** Install queue docked to the bottom of the detail pane: a summary bar collapsed, a table expanded. */
export const InstallQueueBar = () => {
  const { t } = useTranslation();
  const notify = useNotify();
  const contentId = useId();
  const bulkMenuIds = useTooltipTriggerIds();
  const error = useInstallsSelector((snapshot) => snapshot.error);
  const jobs = useInstallsSelector((snapshot) => snapshot.jobs);
  const dismissedJobIds = useInstallsSelector((snapshot) => snapshot.dismissedJobIds);
  const status = useInstallsSelector((snapshot) => snapshot.status);
  const queueExpanded = useModelsUiSelector((snapshot) => snapshot.queueExpanded);
  const queueMaximized = useModelsUiSelector((snapshot) => snapshot.queueMaximized);

  useMountEffect(() => {
    ensureInstallsLoaded();
  });

  const rows = useMemo(() => buildInstallQueueRows(jobs, dismissedJobIds), [dismissedJobIds, jobs]);
  const summary = useMemo(() => summarizeInstallQueue(rows), [rows]);
  const finishedCount = useMemo(() => countFinishedInstallJobs(jobs), [jobs]);
  const featured = rows[0] && isActiveRowStatus(rows[0].status) ? rows[0] : null;
  const isMaximized = queueExpanded && queueMaximized;

  const pausableJobs = useMemo(() => jobs.filter((job) => job.status === 'downloading'), [jobs]);
  const pausedJobs = useMemo(() => jobs.filter((job) => job.status === 'paused'), [jobs]);
  const cancellableJobs = useMemo(
    () => rows.filter((row) => isActiveRowStatus(row.status) || row.status === 'queued' || row.status === 'paused'),
    [rows]
  ).map((row) => row.job);

  // One busy flag per control so each button tracks its own request.
  const { isBusy: isPruning, run: runPrune } = useScopedAction();
  const { isBusy: isRefreshing, run: runRefresh } = useScopedAction();
  const { isBusy: isPausingAll, run: runPauseAll } = useScopedAction();
  const { isBusy: isResumingAll, run: runResumeAll } = useScopedAction();
  const { isBusy: isCancellingAll, run: runCancelAll } = useScopedAction();

  const runBulk = useCallback(
    (
      run: ReturnType<typeof useScopedAction>['run'],
      call: (id: number, signal: AbortSignal) => Promise<unknown>,
      targets: ModelInstallJob[]
    ) =>
      run(
        async (owner) => {
          // allSettled: one failure must not hide the others' outcomes, and the rest still deserve a fresh list.
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

  const handlePrune = () =>
    runPrune(
      async (owner) => {
        await pruneCompletedModelInstalls(owner.signal);
        assertAccountScopeCurrent(owner);
        await refreshInstalls(owner);
      },
      (message) => notify.error(t('models.pruneFailed'), message)
    );

  return (
    <Flex
      bg="bg.subtle"
      borderTopWidth={1}
      direction="column"
      flex={isMaximized ? '1' : undefined}
      flexShrink={0}
      minH="0"
    >
      <HStack flexWrap="wrap" gap="2" minH="2.5rem" px="3" py="1">
        <Button
          aria-controls={contentId}
          aria-expanded={queueExpanded}
          color="fg.muted"
          flexShrink={0}
          gap="1.5"
          ms="-2"
          ps="2"
          pe="3"
          size="xs"
          variant="ghost"
          _hover={TOGGLE_HOVER}
          onClick={() => setQueueExpanded(!queueExpanded)}
        >
          <Icon as={queueExpanded ? ChevronDownIcon : ChevronUpIcon} boxSize="3.5" />
          <Text color="fg" fontSize="xs" fontWeight="700">
            {t('models.installQueue')}
          </Text>
        </Button>

        {queueExpanded ? (
          <SummaryChips summary={summary} />
        ) : (
          <CollapsedSummary featured={featured} jobCount={rows.length} summary={summary} />
        )}

        {queueExpanded ? (
          <HStack flexShrink={0} gap="1">
            {pausableJobs.length + pausedJobs.length + cancellableJobs.length > 0 ? (
              <Group attached>
                {pausedJobs.length > 0 ? (
                  <Button
                    loading={isResumingAll}
                    size="2xs"
                    variant="ghost"
                    onClick={() => void runBulk(runResumeAll, resumeModelInstall, pausedJobs)}
                  >
                    <Icon as={PlayIcon} boxSize="3" />
                    {t('models.resumeAll')}
                  </Button>
                ) : (
                  <Button
                    disabled={pausableJobs.length === 0}
                    loading={isPausingAll}
                    size="2xs"
                    variant="ghost"
                    onClick={() => void runBulk(runPauseAll, pauseModelInstall, pausableJobs)}
                  >
                    <Icon as={PauseIcon} boxSize="3" />
                    {t('models.pauseAll')}
                  </Button>
                )}
                <Menu.Root ids={bulkMenuIds} positioning={BULK_MENU_POSITIONING}>
                  <Tooltip content={t('models.moreQueueActions')} ids={bulkMenuIds}>
                    <Menu.Trigger asChild>
                      <IconButton aria-label={t('models.moreQueueActions')} size="2xs" variant="ghost">
                        <Icon as={ChevronDownIcon} boxSize="3" />
                      </IconButton>
                    </Menu.Trigger>
                  </Tooltip>
                  <Portal>
                    <Menu.Positioner>
                      <MenuContent minW="10rem">
                        <MenuActionItem
                          disabled={cancellableJobs.length === 0 || isCancellingAll}
                          icon={XIcon}
                          label={t('models.cancelAll')}
                          tone="danger"
                          value="cancel-all"
                          onSelect={() => void runBulk(runCancelAll, cancelModelInstall, cancellableJobs)}
                        />
                      </MenuContent>
                    </Menu.Positioner>
                  </Portal>
                </Menu.Root>
              </Group>
            ) : null}
            {pausableJobs.length + pausedJobs.length + cancellableJobs.length > 0 ? (
              <Separator h="4" mx="1" orientation="vertical" />
            ) : null}
            <Button loading={isRefreshing} size="2xs" variant="ghost" onClick={() => runRefresh(refreshInstalls)}>
              <Icon as={RefreshCcwIcon} boxSize="3" />
              {t('common.refresh')}
            </Button>
            <Button disabled={finishedCount === 0} loading={isPruning} size="2xs" variant="ghost" onClick={handlePrune}>
              <Icon as={Trash2Icon} boxSize="3" />
              {t('models.clearFinished')}
            </Button>
            <Tooltip content={queueMaximized ? t('models.restoreQueueTooltip') : t('models.maximizeQueueTooltip')}>
              <Button size="2xs" variant="ghost" onClick={() => setQueueMaximized(!queueMaximized)}>
                <Icon as={queueMaximized ? Minimize2Icon : Maximize2Icon} boxSize="3" />
                {queueMaximized ? t('models.restoreQueue') : t('models.maximizeQueue')}
              </Button>
            </Tooltip>
          </HStack>
        ) : (
          <HStack flexShrink={0} gap="0.5">
            {featured ? <FeaturedActions row={featured} /> : null}
            {featured && finishedCount > 0 ? <Separator h="4" mx="1" orientation="vertical" /> : null}
            {finishedCount > 0 ? (
              <Tooltip content={t('models.clearFinished')}>
                <IconButton
                  aria-label={t('models.clearFinished')}
                  loading={isPruning}
                  size="2xs"
                  variant="ghost"
                  onClick={handlePrune}
                >
                  <Icon as={Trash2Icon} boxSize="3.5" />
                </IconButton>
              </Tooltip>
            ) : null}
          </HStack>
        )}
      </HStack>

      {queueExpanded ? (
        <Box
          borderTopWidth={1}
          flex={isMaximized ? '1' : undefined}
          h={isMaximized ? undefined : DOCKED_HEIGHT}
          id={contentId}
          minH="0"
        >
          <InstallQueueTable error={error} rows={rows} status={status} />
        </Box>
      ) : null}
    </Flex>
  );
};

const CountChip = ({
  count,
  labelKey,
  palette,
  icon,
}: {
  count: number;
  labelKey: string;
  palette: string;
  icon?: 'dot' | 'alert';
}) => {
  const { t } = useTranslation();

  if (count === 0) {
    return null;
  }

  return (
    <Badge colorPalette={palette} flexShrink={0} fontSize="2xs" gap="1.5" rounded="full" size="sm" variant="surface">
      {icon === 'dot' ? <Box bg="colorPalette.solid" boxSize="1.5" rounded="full" /> : null}
      {icon === 'alert' ? <Icon as={TriangleAlertIcon} boxSize="3" /> : null}
      {t(labelKey, { count })}
    </Badge>
  );
};

// A real basis lets the header actions wrap beneath the chips instead of starving them in narrow panes.
const SummaryChips = ({ summary }: { summary: InstallQueueSummary }) => (
  <HStack flex="1 1 18rem" flexWrap="wrap" gap="1" minW="0">
    <CountChip count={summary.downloading} icon="dot" labelKey="models.queueDownloadingCount" palette="accent" />
    <CountChip count={summary.installing} icon="dot" labelKey="models.queueInstallingCount" palette="accent" />
    <CountChip count={summary.queued} labelKey="models.queueQueuedCount" palette="gray" />
    <CountChip count={summary.paused} labelKey="models.queuePausedCount" palette="gray" />
    <CountChip count={summary.attention} icon="alert" labelKey="models.queueAttentionCount" palette="red" />
    <CountChip count={summary.installed} labelKey="models.queueInstalledCount" palette="green" />
  </HStack>
);

const CollapsedSummary = ({
  featured,
  jobCount,
  summary,
}: {
  featured: InstallQueueRow | null;
  jobCount: number;
  summary: InstallQueueSummary;
}) => {
  const { t } = useTranslation();

  // Chips wrap under the featured job before anything overlaps in a narrow pane.
  return (
    <HStack flex="1 1 16rem" flexWrap="wrap" gap="2" minW="0">
      {featured ? (
        <FeaturedJob row={featured} />
      ) : (
        <Text color="fg.muted" flex="1 1 10rem" fontSize="xs" minW="0" truncate>
          {summary.paused > 0
            ? t('models.queuePausedCount', { count: summary.paused })
            : jobCount > 0
              ? t('models.installJobSummary', { count: jobCount })
              : t('models.noInstallsYet')}
        </Text>
      )}
      <HStack flexShrink={0} gap="1">
        <CountChip count={summary.queued} labelKey="models.queueMoreQueued" palette="gray" />
        <CountChip count={summary.attention} icon="alert" labelKey="models.queueAttentionCount" palette="red" />
      </HStack>
    </HStack>
  );
};

const FeaturedJob = ({ row }: { row: InstallQueueRow }) => {
  const { t } = useTranslation();
  const liveProgress = useInstallProgress(row.job.id);
  const progress = resolveInstallProgress(row.job, liveProgress);
  const caption = useInstallProgressCaption({
    compact: true,
    progress,
    queuePosition: null,
    status: row.status,
    typeLabel: null,
  });

  return (
    <HStack flex="1 1 14rem" gap="2" minW="0" overflow="hidden">
      <Spinner borderWidth="1.5px" boxSize="3.5" color="accent.solid" flexShrink={0} />
      <MiddleTruncate
        flexShrink={1}
        fontSize="xs"
        fontWeight="600"
        maxW="18rem"
        minW="5rem"
        text={getInstallJobDisplayName(row.job)}
      />
      <Box flexShrink={1} maxW="10rem" minW="3rem" w="full">
        <InstallProgressBar label={t('models.downloadProgress')} progress={progress} status={row.status} />
      </Box>
      <Text color="fg.muted" flexShrink={3} fontFamily="mono" fontSize="2xs" minW="0" truncate>
        {caption}
      </Text>
    </HStack>
  );
};

const FeaturedActions = ({ row }: { row: InstallQueueRow }) => {
  const { t } = useTranslation();
  const actions = useInstallJobActions(row.job);

  return (
    <>
      {row.status === 'downloading' ? (
        <Tooltip content={t('models.pauseDownload')}>
          <IconButton
            aria-label={t('models.pauseDownload')}
            disabled={actions.isBusy}
            size="2xs"
            variant="ghost"
            onClick={actions.pause}
          >
            <Icon as={PauseIcon} boxSize="3.5" />
          </IconButton>
        </Tooltip>
      ) : null}
      <Tooltip content={t('models.cancelInstall')}>
        <IconButton
          aria-label={t('models.cancelInstall')}
          colorPalette="danger"
          disabled={actions.isBusy}
          size="2xs"
          variant="ghost"
          onClick={actions.cancel}
        >
          <Icon as={XIcon} boxSize="3.5" />
        </IconButton>
      </Tooltip>
    </>
  );
};
