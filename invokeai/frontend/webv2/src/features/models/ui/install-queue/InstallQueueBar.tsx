import type { ModelInstallJob } from '@features/models/core/types';

import { Icon, Menu, Portal, Separator } from '@chakra-ui/react';
import {
  cancelModelInstall,
  pauseModelInstall,
  pruneCompletedModelInstalls,
  resumeModelInstall,
} from '@features/models/data/api';
import {
  ensureInstallsLoaded,
  dismissInstallJob,
  dismissInstallJobs,
  projectInstallCancelled,
  refreshInstalls,
  replaceInstallJob,
  useInstallProgress,
  useInstallsSelector,
} from '@features/models/data/installsStore';
import { setQueueExpanded, setQueueMaximized, useModelsUiSelector } from '@features/models/ui/uiStore';
import { useNotify } from '@features/models/ui/useModelsNotify';
import { useMountEffect } from '@platform/react/useMountEffect';
import { getErrorMessage, useScopedAction } from '@platform/react/useScopedAction';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { Button, IconButton } from '@platform/ui/Button';
import { Group } from '@platform/ui/Group';
import {
  ManagerQueueChip,
  ManagerQueueFeatured,
  ManagerQueueFooter,
  ManagerQueueSummaryText,
} from '@platform/ui/ManagerQueue';
import { MenuActionItem, MenuContent } from '@platform/ui/Menu';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { ChevronDownIcon, PauseIcon, PlayIcon, RefreshCcwIcon, XIcon } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { InstallQueueRow, InstallQueueSummary } from './queueModel';

import { getInstallProgressValue, useInstallProgressCaption } from './InstallProgressCell';
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

const BULK_MENU_POSITIONING = { placement: 'bottom-start' } as const;

/** Install queue docked to the bottom of the detail pane: a summary bar collapsed, a table expanded. */
export const InstallQueueBar = () => {
  const { t } = useTranslation();
  const notify = useNotify();
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
      call: (id: number, signal: AbortSignal) => Promise<ModelInstallJob | null>,
      targets: ModelInstallJob[],
      onSucceeded?: (id: number, result: ModelInstallJob | null) => void
    ) =>
      run(
        async (owner) => {
          // allSettled: one failure must not hide the others' outcomes, and the rest still deserve a fresh list.
          const results = await Promise.allSettled(targets.map((job) => call(job.id, owner.signal)));
          assertAccountScopeCurrent(owner);
          results.forEach((result, index) => {
            if (result.status === 'fulfilled') {
              onSucceeded?.(targets[index]!.id, result.value);
            }
          });
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
        const prunedJobIds = await pruneCompletedModelInstalls(owner.signal);
        assertAccountScopeCurrent(owner);
        dismissInstallJobs(prunedJobIds);
        await refreshInstalls(owner);
      },
      (message) => {
        notify.error(t('models.pruneFailed'), message);
        void refreshInstalls();
      }
    );

  const hasBulkTargets = pausableJobs.length + pausedJobs.length + cancellableJobs.length > 0;

  return (
    <ManagerQueueFooter
      collapsedActions={featured ? <FeaturedActions row={featured} /> : null}
      collapsedChips={
        <>
          <ManagerQueueChip
            count={summary.queued}
            label={t('models.queueMoreQueued', { count: summary.queued })}
            palette="gray"
          />
          <ManagerQueueChip
            count={summary.attention}
            icon="alert"
            label={t('models.queueAttentionCount', { count: summary.attention })}
            palette="red"
          />
        </>
      }
      collapsedSummary={
        featured ? (
          <FeaturedJob row={featured} />
        ) : (
          <ManagerQueueSummaryText>
            {summary.paused > 0
              ? t('models.queuePausedCount', { count: summary.paused })
              : rows.length > 0
                ? t('models.installJobSummary', { count: rows.length })
                : t('models.noInstallsYet')}
          </ManagerQueueSummaryText>
        )
      }
      expanded={queueExpanded}
      expandedActions={
        <>
          {hasBulkTargets ? (
            <Group attached>
              {pausedJobs.length > 0 ? (
                <Button
                  loading={isResumingAll}
                  size="sm"
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
                  size="sm"
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
                    <IconButton aria-label={t('models.moreQueueActions')} size="sm" variant="ghost">
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
                        onSelect={() =>
                          void runBulk(runCancelAll, cancelModelInstall, cancellableJobs, (id, result) => {
                            if (result === null) {
                              dismissInstallJob(id);
                            } else {
                              replaceInstallJob(result);
                              if (!['completed', 'error', 'cancelled'].includes(result.status)) {
                                projectInstallCancelled(id);
                              }
                            }
                          })
                        }
                      />
                    </MenuContent>
                  </Menu.Positioner>
                </Portal>
              </Menu.Root>
            </Group>
          ) : null}
          {hasBulkTargets ? <Separator h="4" mx="1" orientation="vertical" /> : null}
          <Button loading={isRefreshing} size="sm" variant="ghost" onClick={() => runRefresh(refreshInstalls)}>
            <Icon as={RefreshCcwIcon} boxSize="3" />
            {t('common.refresh')}
          </Button>
        </>
      }
      expandedSummary={<SummaryChips summary={summary} />}
      finishedCount={finishedCount}
      isClearingFinished={isPruning}
      maximizeTooltip={t('models.maximizeQueueTooltip')}
      maximized={queueMaximized}
      restoreTooltip={t('models.restoreQueueTooltip')}
      title={t('models.installQueue')}
      onClearFinished={handlePrune}
      onExpandedChange={setQueueExpanded}
      onMaximizedChange={setQueueMaximized}
    >
      <InstallQueueTable error={error} rows={rows} status={status} />
    </ManagerQueueFooter>
  );
};

const SummaryChips = ({ summary }: { summary: InstallQueueSummary }) => {
  const { t } = useTranslation();

  return (
    <>
      <ManagerQueueChip
        count={summary.downloading}
        icon="dot"
        label={t('models.queueDownloadingCount', { count: summary.downloading })}
        palette="accent"
      />
      <ManagerQueueChip
        count={summary.installing}
        icon="dot"
        label={t('models.queueInstallingCount', { count: summary.installing })}
        palette="accent"
      />
      <ManagerQueueChip
        count={summary.queued}
        label={t('models.queueQueuedCount', { count: summary.queued })}
        palette="gray"
      />
      <ManagerQueueChip
        count={summary.paused}
        label={t('models.queuePausedCount', { count: summary.paused })}
        palette="gray"
      />
      <ManagerQueueChip
        count={summary.attention}
        icon="alert"
        label={t('models.queueAttentionCount', { count: summary.attention })}
        palette="red"
      />
      <ManagerQueueChip
        count={summary.installed}
        label={t('models.queueInstalledCount', { count: summary.installed })}
        palette="green"
      />
    </>
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
    <ManagerQueueFeatured
      caption={caption}
      name={getInstallJobDisplayName(row.job)}
      progressLabel={t('models.downloadProgress')}
      progressValue={getInstallProgressValue(progress, row.status)}
    />
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
            size="sm"
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
          size="sm"
          variant="ghost"
          onClick={actions.cancel}
        >
          <Icon as={XIcon} boxSize="3.5" />
        </IconButton>
      </Tooltip>
    </>
  );
};
