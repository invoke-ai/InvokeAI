import type { CustomNodeInstallLogEntry } from '@features/nodes/data/installLogStore';

import { Text } from '@chakra-ui/react';
import {
  clearFinishedCustomNodeInstalls,
  removeCustomNodeInstallLogEntry,
  useCustomNodeInstallLog,
} from '@features/nodes/data/installLogStore';
import { useCustomNodesSelector } from '@features/nodes/data/nodesStore';
import {
  openNodePackDetail,
  setNodeQueueExpanded,
  setNodeQueueMaximized,
  useNodesUiSelector,
} from '@features/nodes/ui/nodesUiStore';
import { useNodePackInstall } from '@features/nodes/ui/shared/useNodePackInstall';
import {
  ManagerQueueChip,
  ManagerQueueErrorDetail,
  ManagerQueueFeatured,
  ManagerQueueFooter,
  ManagerQueueProgressBar,
  ManagerQueueRow,
  ManagerQueueRowAction,
  ManagerQueueSummaryText,
  ManagerQueueTable,
  type ManagerQueueGlyph,
} from '@platform/ui/ManagerQueue';
import {
  CheckIcon,
  DownloadIcon,
  MinusIcon,
  RotateCcwIcon,
  SquareArrowOutUpRightIcon,
  TriangleAlertIcon,
  XIcon,
} from 'lucide-react';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */

type EntryStatus = CustomNodeInstallLogEntry['status'];

/**
 * Installs clone synchronously with no job id or progress, so a running install shows an indeterminate bar and
 * offers no pause or cancel; uninstalls are recorded settled.
 */
const STATUS_PRESENTATION: Record<
  EntryStatus,
  { captionKey: string; glyph: ManagerQueueGlyph; labelKey: string; palette: string }
> = {
  completed: {
    captionKey: 'common.done',
    glyph: { color: 'fg.success', icon: CheckIcon },
    labelKey: 'nodes.installed',
    palette: 'green',
  },
  error: {
    captionKey: 'nodes.notInstalled',
    glyph: { color: 'fg.error', icon: TriangleAlertIcon },
    labelKey: 'common.failed',
    palette: 'red',
  },
  installing: { captionKey: 'nodes.cloningRepository', glyph: 'active', labelKey: 'nodes.installing', palette: 'blue' },
  uninstalled: {
    captionKey: 'nodes.restartToApply',
    glyph: { color: 'fg.muted', icon: MinusIcon },
    labelKey: 'nodes.uninstalled',
    palette: 'gray',
  },
};

const countStatus = (log: readonly CustomNodeInstallLogEntry[], status: EntryStatus): number =>
  log.filter((entry) => entry.status === status).length;

/** Custom node install queue docked under the nodes manager; mirrors the model install queue. */
export const NodeInstallQueue = () => {
  const { t } = useTranslation();
  const log = useCustomNodeInstallLog();
  const nodePacks = useCustomNodesSelector((snapshot) => snapshot.nodePacks);
  const queueExpanded = useNodesUiSelector((snapshot) => snapshot.queueExpanded);
  const queueMaximized = useNodesUiSelector((snapshot) => snapshot.queueMaximized);
  const { install, isInstalling } = useNodePackInstall();

  const packNames = useMemo(() => new Set(nodePacks.map((pack) => pack.name)), [nodePacks]);
  const counts = useMemo(
    () => ({
      failed: countStatus(log, 'error'),
      installed: countStatus(log, 'completed'),
      installing: countStatus(log, 'installing'),
      uninstalled: countStatus(log, 'uninstalled'),
    }),
    [log]
  );
  const featured = log.find((entry) => entry.status === 'installing') ?? null;
  const finishedCount = log.length - counts.installing;

  const attentionChip = (
    <ManagerQueueChip
      count={counts.failed}
      icon="alert"
      label={t('nodes.queueAttentionCount', { count: counts.failed })}
      palette="red"
    />
  );
  const installedChip = (
    <ManagerQueueChip
      count={counts.installed}
      label={t('nodes.queueInstalledCount', { count: counts.installed })}
      palette="green"
    />
  );

  return (
    <ManagerQueueFooter
      collapsedChips={
        <>
          {attentionChip}
          {installedChip}
        </>
      }
      collapsedSummary={
        featured ? (
          <ManagerQueueFeatured
            caption={t('nodes.cloningRepository')}
            name={featured.name}
            progressLabel={t('nodes.installProgress')}
            progressValue={null}
          />
        ) : (
          <ManagerQueueSummaryText>
            {log.length > 0 ? t('nodes.installJobSummary', { count: log.length }) : t('nodes.noInstallsYet')}
          </ManagerQueueSummaryText>
        )
      }
      expanded={queueExpanded}
      expandedSummary={
        <>
          <ManagerQueueChip
            count={counts.installing}
            icon="dot"
            label={t('nodes.queueInstallingCount', { count: counts.installing })}
            palette="accent"
          />
          {attentionChip}
          {installedChip}
          <ManagerQueueChip
            count={counts.uninstalled}
            label={t('nodes.queueUninstalledCount', { count: counts.uninstalled })}
            palette="gray"
          />
        </>
      }
      finishedCount={finishedCount}
      maximizeTooltip={t('nodes.maximizeQueueTooltip')}
      maximized={queueMaximized}
      restoreTooltip={t('nodes.restoreQueueTooltip')}
      title={t('nodes.installQueue')}
      onClearFinished={clearFinishedCustomNodeInstalls}
      onExpandedChange={setNodeQueueExpanded}
      onMaximizedChange={setNodeQueueMaximized}
    >
      <ManagerQueueTable
        emptyDescription={t('nodes.noInstallsDescription')}
        emptyIcon={DownloadIcon}
        emptyTitle={t('nodes.noInstallsYet')}
        isEmpty={log.length === 0}
        label={t('nodes.installJobs')}
      >
        {log.map((entry) => (
          <NodeInstallRow
            key={entry.id}
            canOpen={entry.status === 'completed' && packNames.has(entry.name)}
            canRetry={!isInstalling}
            entry={entry}
            onRetry={(source) => install(source, { retryOf: entry.id })}
          />
        ))}
      </ManagerQueueTable>
    </ManagerQueueFooter>
  );
};

const NodeInstallRow = ({
  canOpen,
  canRetry,
  entry,
  onRetry,
}: {
  canOpen: boolean;
  canRetry: boolean;
  entry: CustomNodeInstallLogEntry;
  onRetry: (source: string) => void;
}) => {
  const { t } = useTranslation();
  const { source, status } = entry;
  const presentation = STATUS_PRESENTATION[status];

  return (
    <ManagerQueueRow
      caption={t(presentation.captionKey)}
      data-install-status={status}
      glyph={presentation.glyph}
      name={entry.name}
      nameMuted={status === 'completed' || status === 'uninstalled'}
      progressBar={
        status === 'installing' ? <ManagerQueueProgressBar label={t('nodes.installProgress')} value={null} /> : null
      }
      status={{ label: t(presentation.labelKey), palette: presentation.palette }}
      subtitle={source !== undefined && source !== entry.name ? source : null}
      tone={status === 'error' ? 'error' : undefined}
      details={
        entry.message ? (
          status === 'error' ? (
            <ManagerQueueErrorDetail message={entry.message} />
          ) : (
            <Text color="fg.muted" fontSize="xs" lineClamp={2} overflowWrap="anywhere" pt="0.5">
              {entry.message}
            </Text>
          )
        ) : null
      }
      actions={
        status === 'installing' ? null : (
          <>
            {status === 'error' && source !== undefined ? (
              <ManagerQueueRowAction
                disabled={false}
                icon={RotateCcwIcon}
                label={t('nodes.retryInstall')}
                unavailableReason={canRetry ? undefined : t('nodes.retryWaitsForInstall')}
                onClick={() => onRetry(source)}
              />
            ) : null}
            {canOpen ? (
              <ManagerQueueRowAction
                disabled={false}
                icon={SquareArrowOutUpRightIcon}
                label={t('nodes.openPack')}
                onClick={() => openNodePackDetail(entry.name)}
              />
            ) : null}
            <ManagerQueueRowAction
              disabled={false}
              icon={XIcon}
              label={t('nodes.removeFromList')}
              onClick={() => removeCustomNodeInstallLogEntry(entry.id)}
            />
          </>
        )
      }
    />
  );
};
