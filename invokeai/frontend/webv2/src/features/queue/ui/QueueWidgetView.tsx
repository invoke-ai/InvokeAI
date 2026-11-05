import type { QueueActivity } from '@features/queue/core/queueActivity';

import { Stack, VisuallyHidden } from '@chakra-ui/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { ListSectionHeader } from '@platform/ui/list/ListSectionHeader';
import { RetryBoundary } from '@platform/ui/RetryBoundary';
import { Scrollable } from '@platform/ui/Scrollable';
import { StatusWidgetChip } from '@platform/ui/StatusWidgetChip';
import { CloudOffIcon, ListOrderedIcon, PauseIcon, type LucideIcon } from 'lucide-react';
import { lazy, Suspense, useState, useSyncExternalStore, type ComponentProps, type ComponentType } from 'react';
import { useTranslation } from 'react-i18next';

import type * as LocalRunListExports from './LocalRunList';
import type { QueueFilterId } from './queueFilters';
import type { QueueLocalRun } from './QueueUiContext';

import { CurrentBatchSection } from './NowNextSection';
import { QueueFilterTabs } from './QueueFilterTabs';
import { useShownLocalRuns } from './queueScope';
import { useQueueUi } from './QueueUiContext';
import { usePendingQueueItemReveal } from './queueUiStore';
import { RecentSection } from './RecentSection';
import { useQueueWidgetActivity } from './useQueueWidgetActivity';

type LocalRunListModule = Pick<typeof LocalRunListExports, 'LocalRunList'>;

type LocalRunListRows = ComponentType<ComponentProps<typeof LocalRunListExports.LocalRunList>>;

/** An external store of the rows component, so the section renders the fresh one a Retry creates. */
export interface LocalRunListLoader {
  /** The rows, lazily; replaced on Retry after a failure, since `React.lazy` keeps its first rejection. */
  getSnapshot(): { Rows: LocalRunListRows };
  /** Starts the import without waiting; a failed one stays failed until `retry`. */
  preload(): void;
  /** Imports again after a failure (an import in flight or done is reused), for the section's Retry. */
  retry(): Promise<unknown>;
  subscribe(listener: () => void): () => void;
}

/**
 * Loads the Submitting section's rows. Runs the backend has not accepted are rare, so they load with the first one, not
 * with the widget. A failed import stays failed until Retry, so re-rendering never refetches it: the server that
 * refused the run usually serves the chunk too.
 */
export const createLocalRunListLoader = (importer: () => Promise<LocalRunListModule>): LocalRunListLoader => {
  const listeners = new Set<() => void>();
  let failed = false;
  let rows: Promise<{ default: typeof LocalRunListExports.LocalRunList }> | null = null;
  const load = () => {
    if (!rows) {
      const next = importer().then((module) => ({ default: module.LocalRunList }));
      rows = next;
      next.catch(() => {
        if (rows === next) {
          failed = true;
        }
      });
    }

    return rows;
  };
  let snapshot: { Rows: LocalRunListRows } = { Rows: lazy(load) };

  return {
    getSnapshot: () => snapshot,
    preload: () => {
      void load().catch(() => undefined);
    },
    retry: () => {
      if (failed) {
        failed = false;
        rows = null;
        snapshot = { Rows: lazy(load) };
        listeners.forEach((listener) => listener());
      }

      return load();
    },
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
};

const localRunList = createLocalRunListLoader(() => import('./LocalRunList'));

export const QueueWidgetView = ({
  presentation,
  region,
}: {
  presentation?: 'compact' | 'expanded' | 'tooltip';
  region: 'bottom' | 'center' | 'dialog' | 'floating' | 'left' | 'popover' | 'right';
}) => {
  if (region === 'bottom' && presentation !== 'expanded') {
    return <QueueStatusChip />;
  }

  return <QueueContent />;
};

const QueueContent = () => {
  const { t } = useTranslation();
  const [filter, setFilter] = useState<QueueFilterId>('all');
  const revealRequest = usePendingQueueItemReveal();
  const { localRuns } = useQueueUi();
  const shownLocalRuns = useShownLocalRuns();
  const [handledRevealRequestId, setHandledRevealRequestId] = useState<number | null>(null);

  // Adjust the active tab during render so it cannot hide a requested item.
  if (revealRequest !== null && revealRequest.requestId !== handledRevealRequestId) {
    setHandledRevealRequestId(revealRequest.requestId);
    setFilter('all');
  }

  // The counted filters stay put while the item lists scroll on their own.
  return (
    <Stack gap="3" h="full" minH="0" pt="3">
      {localRuns.length > 0 ? <LocalRunListPreload rows={localRunList} /> : null}
      <Stack flexShrink={0} gap="3" px="3">
        <QueueFilterTabs localRunCount={shownLocalRuns.length} value={filter} onChange={setFilter} />
      </Stack>
      <Scrollable flex="1" label={t('widgets.queue.items')} minH="0" overflowX="hidden">
        <Stack gap="3" pb="3" px="3">
          <LocalRunsSection filter={filter} rows={localRunList} runs={shownLocalRuns} />
          <CurrentBatchSection filter={filter} revealRequest={revealRequest} />
          <RecentSection filter={filter} revealRequest={revealRequest} />
        </Stack>
      </Scrollable>
    </Stack>
  );
};

/**
 * Fetches the rows with the first unaccepted run, shown or not, while the server that serves them likely still answers.
 * Mounted while there is one; mounting again later reuses the import, or keeps a failed one failed until Retry.
 */
const LocalRunListPreload = ({ rows }: { rows: LocalRunListLoader }) => {
  useMountEffect(rows.preload);
  return null;
};

/**
 * SUBMITTING — this tab's runs the backend has not accepted yet: being sent, or retried after failures. The backend has
 * no row for them, so this is the one place to see why they wait and to cancel them. Like the current batch, they are
 * unfinished, so they show under All and Active alone. The header shows at once, so the section does not jump when its
 * rows arrive; rows that fail to load leave the rest of the widget working.
 */
export const LocalRunsSection = ({
  filter,
  rows,
  runs,
}: {
  filter: QueueFilterId;
  rows: LocalRunListLoader;
  runs: readonly QueueLocalRun[];
}) => {
  const { t } = useTranslation();
  const current = useSyncExternalStore(rows.subscribe, rows.getSnapshot, rows.getSnapshot);
  const isListed = runs.length > 0 && (filter === 'all' || filter === 'active');

  return (
    <>
      <LocalRunAnnouncements runs={runs} />
      {isListed ? (
        // Row surfaces bleed into the widget padding so their content lines up with the controls above.
        <Stack gap="1" mx="-2">
          <ListSectionHeader count={runs.length} label={t('widgets.queue.localRuns.title')} />
          <RetryBoundary
            message={t('widgets.queue.localRuns.loadFailed')}
            retry={rows.retry}
            retryLabel={t('common.retry')}
          >
            <Suspense fallback={null}>
              <current.Rows filter={filter} runs={runs} />
            </Suspense>
          </RetryBoundary>
        </Stack>
      ) : null}
    </>
  );
};

const RETRY_ANNOUNCEMENT_KEYS = {
  error: 'widgets.queue.localRuns.announcement.error',
  'server-error': 'widgets.queue.localRuns.announcement.serverError',
  unavailable: 'widgets.queue.localRuns.announcement.unavailable',
  unreachable: 'widgets.queue.localRuns.announcement.unreachable',
} satisfies Record<NonNullable<QueueLocalRun['retry']>['reason'], string>;

/**
 * Tells screen readers why a run retries. Each retried run has one line keyed by its reason and naming no attempt, so a
 * line is added, and spoken, only when a run starts retrying or its reason changes, never on another attempt. It
 * stays mounted, whatever the filter, because a region inserted with its text is not announced, and switching tabs is
 * no news. Being offline, and the final failure that removes the run, are left to their own notices.
 */
const LocalRunAnnouncements = ({ runs }: { runs: readonly QueueLocalRun[] }) => {
  const { t } = useTranslation();

  return (
    <VisuallyHidden aria-atomic="false" aria-live="polite" as="div" role="status">
      {runs.map((run) =>
        run.retry ? (
          <div key={`${run.projectId}:${run.id}:${run.retry.reason}`}>
            {t(RETRY_ANNOUNCEMENT_KEYS[run.retry.reason], {
              name: run.label || t('widgets.queue.localRuns.untitled'),
            })}
          </div>
        ) : null
      )}
    </VisuallyHidden>
  );
};

const chipIcons = {
  idle: ListOrderedIcon,
  'loading-model': ListOrderedIcon,
  offline: CloudOffIcon,
  paused: PauseIcon,
  running: ListOrderedIcon,
} satisfies Record<QueueActivity, LucideIcon>;

const QueueStatusChip = () => {
  const { t } = useTranslation();
  const { activity, inProgress, loadingModel, pending } = useQueueWidgetActivity();
  const openCount = inProgress + pending;
  const label =
    activity === 'offline'
      ? t('widgets.queueStatus.offline', { count: openCount })
      : activity === 'paused'
        ? t('widgets.queueStatus.paused', { count: openCount })
        : activity === 'loading-model'
          ? t('widgets.queue.loadingModel', { model: loadingModel })
          : inProgress > 0
            ? t('widgets.queue.generating', { count: inProgress })
            : t('widgets.queue.queued', { count: pending });

  return (
    <StatusWidgetChip
      icon={chipIcons[activity]}
      tone={
        activity === 'offline'
          ? 'warning'
          : activity === 'running' || activity === 'loading-model'
            ? 'accent'
            : undefined
      }
    >
      {label}
    </StatusWidgetChip>
  );
};
