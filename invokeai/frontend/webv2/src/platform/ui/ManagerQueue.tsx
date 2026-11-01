import type { LucideIcon } from 'lucide-react';
import type { ReactNode } from 'react';

import {
  Badge,
  Box,
  Flex,
  HStack,
  Icon,
  Progress,
  Separator,
  Spinner,
  Stack,
  Table,
  Text,
  VisuallyHidden,
  type SystemStyleObject,
} from '@chakra-ui/react';
import {
  ChevronDownIcon,
  ChevronUpIcon,
  Maximize2Icon,
  Minimize2Icon,
  Trash2Icon,
  TriangleAlertIcon,
} from 'lucide-react';
import { useId } from 'react';
import { useTranslation } from 'react-i18next';

/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop */
import { Button, IconButton } from './Button';
import { EmptyState } from './EmptyState';
import { MiddleTruncate } from './MiddleTruncate';
import { Scrollable } from './Scrollable';
import { Tooltip } from './Tooltip';

/**
 * Presentational frame for a manager's job queue docked under its panes: the footer bar, summary chips, and the job
 * table. Owners keep the job model, status vocabulary, and actions; these pieces only lay them out.
 */

const DOCKED_HEIGHT = 'min(30rem, 60dvh)';
const TOGGLE_HOVER = { color: 'fg' } as const;

const TABLE_SX: SystemStyleObject = {
  '& th': { bg: 'bg.subtle', position: 'sticky', top: 0, zIndex: 1 },
  '& tr[data-tone="error"] > td': { bg: 'bg.error' },
  '& tr[data-tone="error"] > td:first-of-type': { boxShadow: 'inset 2px 0 0 {colors.border.error}' },
  '& tr[data-tone="warning"] > td': { bg: 'bg.warning' },
  '& tr[data-tone="warning"] > td:first-of-type': { boxShadow: 'inset 2px 0 0 {colors.fg.warning}' },
};

const HEADER_PROPS = {
  borderColor: 'border.subtle',
  color: 'fg.muted',
  fontSize: 'xs',
  fontWeight: '700',
  letterSpacing: 'wider',
  py: '1.5',
  textTransform: 'uppercase',
} as const;

const CELL_PROPS = { borderColor: 'border.subtle', py: '2', verticalAlign: 'top' } as const;

export interface ManagerQueueFooterProps {
  title: string;
  expanded: boolean;
  onExpandedChange: (expanded: boolean) => void;
  maximized: boolean;
  onMaximizedChange: (maximized: boolean) => void;
  maximizeTooltip: string;
  restoreTooltip: string;
  /** Chips shown beside the title while expanded. */
  expandedSummary: ReactNode;
  /** The featured job, or a summary line, while collapsed. */
  collapsedSummary: ReactNode;
  /** Chips that follow the collapsed summary. */
  collapsedChips?: ReactNode;
  /** Owner actions ahead of Clear finished while expanded. */
  expandedActions?: ReactNode;
  /** Actions for the featured job while collapsed. */
  collapsedActions?: ReactNode;
  /** Settled jobs Clear finished would remove; the control disables (expanded) or hides (collapsed) at zero. */
  finishedCount: number;
  isClearingFinished?: boolean;
  onClearFinished: () => void;
  /** The job table, shown while expanded. */
  children: ReactNode;
}

/** Queue docked to the bottom of a manager: a summary bar collapsed, a table expanded, or the whole pane maximized. */
export const ManagerQueueFooter = ({
  children,
  collapsedActions,
  collapsedChips,
  collapsedSummary,
  expanded,
  expandedActions,
  expandedSummary,
  finishedCount,
  isClearingFinished = false,
  maximizeTooltip,
  maximized,
  onClearFinished,
  onExpandedChange,
  onMaximizedChange,
  restoreTooltip,
  title,
}: ManagerQueueFooterProps) => {
  const { t } = useTranslation();
  const contentId = useId();
  const isMaximized = expanded && maximized;

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
          aria-expanded={expanded}
          color="fg.muted"
          flexShrink={0}
          gap="1.5"
          ms="-2"
          ps="2"
          pe="3"
          variant="ghost"
          _hover={TOGGLE_HOVER}
          onClick={() => onExpandedChange(!expanded)}
        >
          <Icon as={expanded ? ChevronDownIcon : ChevronUpIcon} boxSize="3.5" />
          <Text color="fg" fontWeight="700">
            {title}
          </Text>
        </Button>

        {expanded ? (
          // A real basis lets the header actions wrap beneath the chips instead of starving them in narrow panes.
          <HStack flex="1 1 18rem" flexWrap="wrap" gap="1" minW="0">
            {expandedSummary}
          </HStack>
        ) : (
          // Chips wrap under the featured job before anything overlaps in a narrow pane.
          <HStack flex="1 1 16rem" flexWrap="wrap" gap="2" minW="0">
            {collapsedSummary}
            <HStack flexShrink={0} gap="1">
              {collapsedChips}
            </HStack>
          </HStack>
        )}

        {expanded ? (
          <HStack flexShrink={0} gap="1">
            {expandedActions}
            <Button
              disabled={finishedCount === 0}
              loading={isClearingFinished}
              size="sm"
              variant="ghost"
              onClick={onClearFinished}
            >
              <Icon as={Trash2Icon} boxSize="3" />
              {t('common.managerQueue.clearFinished')}
            </Button>
            <Tooltip content={maximized ? restoreTooltip : maximizeTooltip}>
              <Button size="sm" variant="ghost" onClick={() => onMaximizedChange(!maximized)}>
                <Icon as={maximized ? Minimize2Icon : Maximize2Icon} boxSize="3" />
                {maximized ? t('common.managerQueue.restore') : t('common.managerQueue.maximize')}
              </Button>
            </Tooltip>
          </HStack>
        ) : (
          <HStack flexShrink={0} gap="0.5">
            {collapsedActions}
            {collapsedActions && finishedCount > 0 ? <Separator h="4" mx="1" orientation="vertical" /> : null}
            {finishedCount > 0 ? (
              <Tooltip content={t('common.managerQueue.clearFinished')}>
                <IconButton
                  aria-label={t('common.managerQueue.clearFinished')}
                  loading={isClearingFinished}
                  size="sm"
                  variant="ghost"
                  onClick={onClearFinished}
                >
                  <Icon as={Trash2Icon} boxSize="3.5" />
                </IconButton>
              </Tooltip>
            ) : null}
          </HStack>
        )}
      </HStack>

      {expanded ? (
        <Box
          borderTopWidth={1}
          flex={isMaximized ? '1' : undefined}
          h={isMaximized ? undefined : DOCKED_HEIGHT}
          id={contentId}
          minH="0"
        >
          {children}
        </Box>
      ) : null}
    </Flex>
  );
};

/** A count of jobs in one state; hidden at zero. `dot` marks live work, `alert` work needing the user. */
export const ManagerQueueChip = ({
  count,
  icon,
  label,
  palette,
}: {
  count: number;
  icon?: 'dot' | 'alert';
  label: string;
  palette: string;
}) => {
  if (count === 0) {
    return null;
  }

  return (
    <Badge colorPalette={palette} flexShrink={0} fontSize="xs" gap="1.5" rounded="full" size="lg" variant="surface">
      {icon === 'dot' ? <Box bg="colorPalette.solid" boxSize="1.5" rounded="full" /> : null}
      {icon === 'alert' ? <Icon as={TriangleAlertIcon} boxSize="3" /> : null}
      {label}
    </Badge>
  );
};

/** Collapsed summary line when no job is featured. */
export const ManagerQueueSummaryText = ({ children }: { children: string }) => (
  <Text color="fg.muted" flex="1 1 10rem" minW="0" truncate>
    {children}
  </Text>
);

/** `value` is a 0–1 ratio; null sweeps for work with no measurable progress. */
export const ManagerQueueProgressBar = ({
  label,
  muted = false,
  value,
}: {
  label: string;
  muted?: boolean;
  value: number | null;
}) => (
  <Progress.Root aria-label={label} colorPalette={muted ? 'gray' : 'accent'} max={1} minW="0" value={value} w="full">
    <Progress.Track rounded="full">
      <Progress.Range rounded="full" transition="width var(--wb-motion-duration-fast) ease" />
    </Progress.Track>
  </Progress.Root>
);

/** The active job the collapsed bar leads with. */
export const ManagerQueueFeatured = ({
  caption,
  name,
  progressLabel,
  progressValue,
}: {
  caption: string;
  name: string;
  progressLabel: string;
  progressValue: number | null;
}) => (
  <HStack flex="1 1 14rem" gap="2" minW="0" overflow="hidden">
    <Spinner borderWidth="1.5px" boxSize="3.5" color="accent.solid" flexShrink={0} />
    <MiddleTruncate flexShrink={1} fontSize="md" fontWeight="600" maxW="18rem" minW="5rem" text={name} />
    <Box flexShrink={1} maxW="10rem" minW="3rem" w="full">
      <ManagerQueueProgressBar label={progressLabel} value={progressValue} />
    </Box>
    <Text color="fg.muted" flexShrink={3} fontFamily="mono" fontSize="xs" minW="0" truncate>
      {caption}
    </Text>
  </HStack>
);

/** `active` spins for running work; settled states name an icon and its color token. */
export type ManagerQueueGlyph = 'active' | { icon: LucideIcon; color: string };

const ManagerQueueStatusGlyph = ({ glyph }: { glyph: ManagerQueueGlyph }) =>
  glyph === 'active' ? (
    <Spinner borderWidth="1.5px" boxSize="3.5" color="accent.solid" />
  ) : (
    <Icon as={glyph.icon} boxSize="3.5" color={glyph.color} />
  );

export const ManagerQueueRowAction = ({
  disabled,
  icon,
  label,
  onClick,
  tone,
  unavailableReason,
}: {
  disabled: boolean;
  icon: LucideIcon;
  label: string;
  onClick: () => void;
  tone?: 'danger';
  /** Why the action cannot run right now; unlike `disabled`, it stays focusable and explains itself. */
  unavailableReason?: string;
}) => (
  <IconButton
    aria-label={label}
    colorPalette={tone === 'danger' ? 'danger' : undefined}
    disabled={disabled}
    size="sm"
    tooltip={label}
    unavailableReason={unavailableReason}
    variant="ghost"
    onClick={onClick}
  >
    <Icon as={icon} boxSize="3" />
  </IconButton>
);

/** A failure under the job name: the backend's reason code, when it has one, beside its message. */
export const ManagerQueueErrorDetail = ({ message, reason }: { message: string; reason?: string | null }) => (
  <HStack align="baseline" flexWrap="wrap" gap="1.5" pt="0.5">
    {reason ? (
      <Badge colorPalette="red" fontFamily="mono" size="lg" variant="surface">
        {reason}
      </Badge>
    ) : null}
    <Text color="fg" fontSize="xs" overflowWrap="anywhere">
      {message}
    </Text>
  </HStack>
);

export interface ManagerQueueRowProps extends Omit<Table.RowProps, 'children'> {
  glyph: ManagerQueueGlyph;
  name: string;
  /** Settled successes recede so live and failed jobs stand out. */
  nameMuted?: boolean;
  /** Mono source line under the name; omitted when null. */
  subtitle?: string | null;
  /** Owner content under the name: errors, fixes, per-file problems. */
  details?: ReactNode;
  progressBar?: ReactNode;
  caption: string;
  status: { label: string; palette: string };
  /** Beside the status badge, e.g. a stale-progress warning. */
  statusAdornment?: ReactNode;
  tone?: 'error' | 'warning';
  actions?: ReactNode;
}

export const ManagerQueueRow = ({
  actions,
  caption,
  details,
  glyph,
  name,
  nameMuted = false,
  progressBar,
  status,
  statusAdornment,
  subtitle = null,
  tone,
  ...rowProps
}: ManagerQueueRowProps) => (
  <Table.Row data-tone={tone} {...rowProps}>
    <Table.Cell {...CELL_PROPS} ps="3" pt="2.5">
      <ManagerQueueStatusGlyph glyph={glyph} />
    </Table.Cell>

    <Table.Cell {...CELL_PROPS}>
      <Stack gap="0" minW="0">
        <MiddleTruncate
          color={nameMuted ? 'fg.muted' : 'fg'}
          fontSize="md"
          fontWeight="600"
          lineHeight="short"
          text={name}
        />
        {subtitle !== null ? <MiddleTruncate color="fg.muted" fontFamily="mono" fontSize="xs" text={subtitle} /> : null}
        {details}
      </Stack>
    </Table.Cell>

    <Table.Cell {...CELL_PROPS}>
      <Stack gap="1" minW="0">
        {progressBar}
        <Text color="fg.muted" fontFamily="mono" fontSize="xs" lineHeight="short" overflowWrap="anywhere">
          {caption}
        </Text>
      </Stack>
    </Table.Cell>

    <Table.Cell {...CELL_PROPS}>
      <HStack gap="1.5">
        <Badge
          colorPalette={status.palette}
          fontSize="xs"
          fontWeight="700"
          letterSpacing="wider"
          size="lg"
          textTransform="uppercase"
          variant="surface"
        >
          {status.label}
        </Badge>
        {statusAdornment}
      </HStack>
    </Table.Cell>

    <Table.Cell {...CELL_PROPS} pe="2">
      <HStack gap="0.5" justify="flex-end">
        {actions}
      </HStack>
    </Table.Cell>
  </Table.Row>
);

export interface ManagerQueueTableProps {
  /** Accessible name of the scrolling job list. */
  label: string;
  isLoading?: boolean;
  /** Replaces the table when the queue itself could not load. */
  loadError?: { title: string; description: string | null; onRetry: () => void } | null;
  isEmpty: boolean;
  emptyIcon: LucideIcon;
  emptyTitle: string;
  emptyDescription: string;
  /** `ManagerQueueRow`s. */
  children: ReactNode;
}

export const ManagerQueueTable = ({
  children,
  emptyDescription,
  emptyIcon,
  emptyTitle,
  isEmpty,
  isLoading = false,
  label,
  loadError = null,
}: ManagerQueueTableProps) => {
  const { t } = useTranslation();

  if (isLoading) {
    return (
      <Flex align="center" h="full" justify="center" py="6">
        <Spinner color="fg.muted" size="lg" />
      </Flex>
    );
  }

  if (loadError) {
    return (
      <EmptyState danger description={loadError.description ?? undefined} h="full" title={loadError.title}>
        <Button variant="outline" onClick={loadError.onRetry}>
          {t('common.retry')}
        </Button>
      </EmptyState>
    );
  }

  if (isEmpty) {
    return <EmptyState description={emptyDescription} h="full" icon={<Icon as={emptyIcon} />} title={emptyTitle} />;
  }

  return (
    <Scrollable h="full" label={label} minH="0">
      <Table.Root css={TABLE_SX} minW="32rem" tableLayout="fixed" w="full">
        <Table.ColumnGroup>
          <Table.Column w="2.25rem" />
          <Table.Column />
          <Table.Column w="30%" />
          <Table.Column w="7.75rem" />
          <Table.Column w="4.5rem" />
        </Table.ColumnGroup>
        <Table.Header>
          <Table.Row>
            <Table.ColumnHeader {...HEADER_PROPS} ps="3">
              <VisuallyHidden>{t('common.managerQueue.state')}</VisuallyHidden>
            </Table.ColumnHeader>
            <Table.ColumnHeader {...HEADER_PROPS}>{t('common.managerQueue.name')}</Table.ColumnHeader>
            <Table.ColumnHeader {...HEADER_PROPS}>{t('common.managerQueue.progress')}</Table.ColumnHeader>
            <Table.ColumnHeader {...HEADER_PROPS}>{t('common.managerQueue.status')}</Table.ColumnHeader>
            <Table.ColumnHeader {...HEADER_PROPS} pe="2" textAlign="end">
              {t('common.managerQueue.actions')}
            </Table.ColumnHeader>
          </Table.Row>
        </Table.Header>
        <Table.Body>{children}</Table.Body>
      </Table.Root>
    </Scrollable>
  );
};
