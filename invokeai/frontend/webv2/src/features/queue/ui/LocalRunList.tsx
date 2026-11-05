/* eslint-disable react-perf/jsx-no-jsx-as-prop */
import type { MouseEvent as ReactMouseEvent } from 'react';

import { Box, Icon, Text } from '@chakra-ui/react';
import { IconButton } from '@platform/ui/Button';
import { ListItem } from '@platform/ui/list/ListItem';
import { ListStack } from '@platform/ui/list/ListStack';
import { Scrollable } from '@platform/ui/Scrollable';
import { segmentTabsTabId } from '@platform/ui/SegmentTabs';
import {
  ChevronRightIcon,
  CircleAlertIcon,
  CloudOffIcon,
  HourglassIcon,
  SendIcon,
  TriangleAlertIcon,
  UnplugIcon,
  XIcon,
  type LucideIcon,
} from 'lucide-react';
import { memo, useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { QueueFilterId } from './queueFilters';

import { QUEUE_FILTER_TABS_ID } from './QueueFilterTabs';
import { useQueueUi, type QueueLocalRun } from './QueueUiContext';

const CHEVRON_OPEN = { transform: 'rotate(90deg)' } as const;

type LocalRunState = NonNullable<QueueLocalRun['retry']>['reason'] | 'offline' | 'submitting';

const STATE_META: Record<LocalRunState, { color: string; icon: LucideIcon; labelKey: string }> = {
  error: { color: 'fg.warning', icon: CircleAlertIcon, labelKey: 'widgets.queue.localRuns.state.error' },
  offline: { color: 'fg.warning', icon: CloudOffIcon, labelKey: 'widgets.queue.localRuns.state.offline' },
  'server-error': {
    color: 'fg.warning',
    icon: TriangleAlertIcon,
    labelKey: 'widgets.queue.localRuns.state.serverError',
  },
  submitting: { color: 'fg.muted', icon: SendIcon, labelKey: 'widgets.queue.localRuns.state.submitting' },
  unavailable: { color: 'fg.warning', icon: HourglassIcon, labelKey: 'widgets.queue.localRuns.state.unavailable' },
  unreachable: { color: 'fg.warning', icon: UnplugIcon, labelKey: 'widgets.queue.localRuns.state.unreachable' },
};

/** Offline, the run waits for the connection whatever its last failure was. */
const getLocalRunState = (run: QueueLocalRun, isConnected: boolean): LocalRunState =>
  !isConnected ? 'offline' : (run.retry?.reason ?? 'submitting');

/**
 * The rows of the Submitting section (`LocalRunsSection` in `QueueWidgetView`), loaded only once this tab has a run the backend has not
 * accepted. The canceled row leaves at once; focus moves to a neighbour's Cancel, else to the filter tabs above, never
 * the page.
 */
export const LocalRunList = ({ filter, runs }: { filter: QueueFilterId; runs: readonly QueueLocalRun[] }) => {
  const { t } = useTranslation();
  const { cancelLocalRun } = useQueueUi();

  const cancel = useCallback(
    (run: QueueLocalRun, button: HTMLElement) => {
      const buttons = [
        ...(button.closest('[role="list"]')?.querySelectorAll<HTMLElement>('[data-local-run-cancel]') ?? []),
      ];
      const index = buttons.indexOf(button);
      const next =
        buttons[index + 1] ??
        buttons[index - 1] ??
        document.getElementById(segmentTabsTabId(QUEUE_FILTER_TABS_ID, filter));

      cancelLocalRun(run.projectId, run.id);
      next?.focus();
    },
    [cancelLocalRun, filter]
  );

  return (
    <ListStack dividers label={t('widgets.queue.localRuns.title')}>
      {runs.map((run) => (
        <LocalRunRow key={`${run.projectId}:${run.id}`} run={run} onCancel={cancel} />
      ))}
    </ListStack>
  );
};

const LocalRunRow = memo(
  ({ onCancel, run }: { onCancel: (run: QueueLocalRun, button: HTMLElement) => void; run: QueueLocalRun }) => {
    const { t } = useTranslation();
    const { isConnected } = useQueueUi();
    const [expanded, setExpanded] = useState(false);
    const toggle = useCallback(() => setExpanded((open) => !open), []);
    const state = getLocalRunState(run, isConnected);
    const meta = STATE_META[state];
    const title = run.label || t('widgets.queue.localRuns.untitled');
    const cancelLabel = t('widgets.queue.localRuns.cancel', { name: title });
    // The failure's text can be long (a traceback's last line, a proxy's page); the row shows its first line, and
    // expanding it shows the whole in a bounded scroll area, so it never pushes the rest of the queue away.
    const message = run.retry?.message.trim() ?? '';
    const canExpand = message !== '';
    const isExpanded = canExpand && expanded;
    const firstLine = message.split('\n')[0];

    const cancel = useCallback(
      (event: ReactMouseEvent<HTMLButtonElement>) => onCancel(run, event.currentTarget),
      [onCancel, run]
    );

    return (
      <Box
        borderColor={isExpanded ? 'border' : 'transparent'}
        borderWidth={1}
        overflow="hidden"
        role="listitem"
        rounded="md"
      >
        <ListItem
          actions={
            <IconButton
              aria-label={cancelLabel}
              color="fg.muted"
              data-local-run-cancel=""
              size="sm"
              tooltip={cancelLabel}
              variant="ghost"
              onClick={cancel}
            >
              <Icon as={XIcon} boxSize="3.5" />
            </IconButton>
          }
          description={
            // Wraps rather than truncates: it is the row's one statement of why the run waits.
            <Text color={meta.color}>{t(meta.labelKey, { attempt: (run.retry?.failureCount ?? 0) + 1 })}</Text>
          }
          isExpanded={canExpand ? isExpanded : undefined}
          leading={
            <Box
              alignItems="center"
              aria-hidden
              bg="bg.subtle"
              boxSize="8"
              color={meta.color}
              display="flex"
              flexShrink={0}
              justifyContent="center"
              rounded="md"
            >
              <Icon as={meta.icon} boxSize="4" />
            </Box>
          }
          role="presentation"
          title={title}
          titleTruncate="end"
          trailing={
            canExpand ? (
              <Icon
                as={ChevronRightIcon}
                boxSize="4"
                transition="transform var(--wb-motion-duration-fast) ease"
                css={isExpanded ? CHEVRON_OPEN : undefined}
              />
            ) : undefined
          }
          onPress={canExpand ? toggle : undefined}
        />

        {isExpanded ? (
          <Scrollable label={t('widgets.queue.localRuns.errorLabel', { name: title })} maxH="8rem" mb="2" mx="2.5">
            <Text color="fg.muted" fontSize="xs" overflowWrap="anywhere" whiteSpace="pre-wrap">
              {message}
            </Text>
          </Scrollable>
        ) : canExpand ? (
          <Text color="fg.muted" fontSize="xs" pb="2" px="2.5" truncate>
            {firstLine}
          </Text>
        ) : null}
      </Box>
    );
  }
);
