import type {
  GalleryMediaUsageGroup,
  GalleryMediaUsageReport,
  GalleryMediaUsageSettledCheck,
  GalleryMediaUsageSurfaceRow,
} from '@features/gallery/core/mediaUsage';
import type { TFunction } from 'i18next';

import { Flex, HStack, Icon, List, Stack, Text } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { Scrollable } from '@platform/ui/Scrollable';
import { ClippedTooltip } from '@platform/ui/Tooltip';
import { TriangleAlertIcon } from 'lucide-react';
import { useCallback, useId, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import { GalleryMediaUsageChecking } from './GalleryMediaUsagePanel';

const KEY = 'widgets.gallery.deleteInUse';

/**
 * The summary line. It counts what the check counted: the requested items, or for a board the items deleting it
 * takes, which can be fewer than the board shows when it holds media of other accounts. One checked item needs no
 * count.
 */
const getSummary = (t: TFunction, report: GalleryMediaUsageReport): string => {
  const { checkedCount: total, inUseCount: count, onlyImages, scope } = report;

  if (total === 1) {
    return t(`${KEY}.summaryOne`);
  }

  const media = t(onlyImages ? 'widgets.gallery.imageCount' : `${KEY}.itemCount`, { count: total });
  const key = scope === 'board' ? `${KEY}.summaryBoard` : `${KEY}.summary`;

  return count === total ? t(`${key}All`, { media }) : t(key, { count, media });
};

const named = (t: TFunction, key: string, unnamedKey: string, name: string | null): string =>
  name === null ? t(unnamedKey) : t(key, { name });

const getGroupLabel = (t: TFunction, group: GalleryMediaUsageGroup): string => {
  switch (group.kind) {
    case 'openProject':
      return t(`${KEY}.group.openProject`);
    case 'project':
    case 'projectUnchecked':
      return named(t, `${KEY}.group.project`, `${KEY}.group.projectUnnamed`, group.name);
    case 'quarantinedProject':
      return named(t, `${KEY}.group.quarantinedProject`, `${KEY}.group.quarantinedProjectUnnamed`, group.name);
    case 'workflow':
      return named(t, `${KEY}.group.workflow`, `${KEY}.group.workflowUnnamed`, group.name);
    case 'legacyEditor':
      return t(`${KEY}.group.legacyEditor`);
    case 'otherDocument':
      return named(t, `${KEY}.group.otherDocument`, `${KEY}.group.otherDocumentUnnamed`, group.name);
    case 'moreDocuments':
      return t(`${KEY}.group.moreDocuments`, { count: group.documentCount });
    case 'queue':
      return t(`${KEY}.group.queue`, { count: group.jobCount });
    case 'otherAccounts':
      return t(`${KEY}.group.otherAccounts`);
  }
};

/**
 * One line: the source's name, clamped to two lines so a long one cannot push its count out of view, and how many it
 * uses. A clamped name shows in full in a tooltip; assistive technology reads the whole text either way.
 */
const UsageLine = ({ count, label, muted = false }: { count: string | null; label: string; muted?: boolean }) => {
  const labelRef = useRef<HTMLParagraphElement | null>(null);
  const isClipped = useCallback(() => {
    const element = labelRef.current;

    return element !== null && element.scrollHeight > element.clientHeight + 1;
  }, []);

  return (
    <Flex align="baseline" fontSize="md" gap="2" minW="0">
      <ClippedTooltip content={label} isClipped={isClipped}>
        <Text
          ref={labelRef}
          color={muted ? 'fg.muted' : undefined}
          flex="1"
          lineClamp={2}
          minW="0"
          overflowWrap="anywhere"
        >
          {label}
        </Text>
      </ClippedTooltip>
      {count === null ? null : (
        <Text color="fg.muted" flexShrink={0} fontVariantNumeric="tabular-nums" whiteSpace="nowrap">
          {count}
        </Text>
      )}
    </Flex>
  );
};

/** Rows leave out their counts when one item is checked: every row would say "1 item". */
const SurfaceRows = ({
  showCounts,
  surfaces,
}: {
  showCounts: boolean;
  surfaces: readonly GalleryMediaUsageSurfaceRow[];
}) => {
  const { t } = useTranslation();

  return (
    <List.Root gap="0.5" listStyleType="none" ps="3">
      {surfaces.map((row, index) => (
        // Built once per report and never reordered; labels such as workflow names can repeat.
        <List.Item key={index}>
          <UsageLine
            count={showCounts ? t(`${KEY}.itemCount`, { count: row.count }) : null}
            label={t(`${KEY}.surface.${row.surface}`, { name: row.label })}
            muted
          />
        </List.Item>
      ))}
    </List.Root>
  );
};

const UsageGroup = ({ group, showCounts }: { group: GalleryMediaUsageGroup; showCounts: boolean }) => {
  const { t } = useTranslation();
  const count = showCounts && group.kind !== 'moreDocuments' ? t(`${KEY}.itemCount`, { count: group.count }) : null;

  return (
    <List.Item>
      <UsageLine count={count} label={getGroupLabel(t, group)} />
      {group.kind === 'openProject' || group.kind === 'project' ? (
        <SurfaceRows showCounts={showCounts} surfaces={group.surfaces} />
      ) : null}
      {(group.kind === 'openProject' || group.kind === 'project') && group.savedCopyUnchecked ? (
        <Text color="fg.muted" fontSize="sm" ps="3">
          {t(`${KEY}.savedCopyUncheckedHint`)}
        </Text>
      ) : null}
      {group.kind === 'projectUnchecked' ? (
        <Text color="fg.muted" fontSize="sm" ps="3">
          {t(`${KEY}.projectUncheckedHint`)}
        </Text>
      ) : null}
    </List.Item>
  );
};

/** A check that did not (fully) run: a warning, distinct from the muted notes, with a way to run it again. */
const UsageWarning = ({
  isRetryDisabled = false,
  message,
  onRetry,
}: {
  isRetryDisabled?: boolean;
  message: string;
  onRetry?: () => void;
}) => {
  const { t } = useTranslation();

  return (
    <HStack align="start" color="fg.warning" fontSize="sm" gap="2">
      <Icon aria-hidden="true" as={TriangleAlertIcon} boxSize="3.5" flexShrink={0} mt="0.5" />
      <Text flex="1">{message}</Text>
      {onRetry ? (
        <Button disabled={isRetryDisabled} flexShrink={0} size="xs" variant="outline" onClick={onRetry}>
          {t('common.retry')}
        </Button>
      ) : null}
    </HStack>
  );
};

const UsageReport = ({
  isRechecking,
  isRetryDisabled,
  onRetry,
  report,
}: {
  isRechecking: boolean;
  isRetryDisabled: boolean;
  onRetry?: () => void;
  report: GalleryMediaUsageReport;
}) => {
  const { t } = useTranslation();
  const headingId = useId();
  // While a retry runs, the check stands where the warning it answers was; the rest of the report stays.
  const unchecked = isRechecking ? (
    <GalleryMediaUsageChecking />
  ) : report.serverCheck === 'complete' ? null : (
    <UsageWarning
      isRetryDisabled={isRetryDisabled}
      message={t(
        report.serverCheck === 'tooManyItems'
          ? `${KEY}.tooManyItems`
          : report.groups.length === 0
            ? `${KEY}.uncheckedNone`
            : `${KEY}.unchecked`,
        { count: report.checkedCount }
      )}
      // Retrying cannot shrink a selection past the limit.
      onRetry={report.serverCheck === 'unavailable' ? onRetry : undefined}
    />
  );

  if (report.groups.length === 0) {
    return (
      unchecked ?? (
        <Text color="fg.muted" fontSize="sm">
          {t(`${KEY}.none`)}
        </Text>
      )
    );
  }

  const showCounts = report.checkedCount > 1;
  const hasQueue = report.groups.some((group) => group.kind === 'queue');

  return (
    <Stack aria-labelledby={headingId} as="section" gap="1.5" role="group">
      <Text fontSize="md" fontWeight="semibold" id={headingId}>
        {getSummary(t, report)}
      </Text>
      <Scrollable label={t(`${KEY}.listLabel`)} maxH="14rem">
        <List.Root gap="1" listStyleType="none">
          {report.groups.map((group, index) => (
            <UsageGroup key={index} group={group} showCounts={showCounts} />
          ))}
        </List.Root>
      </Scrollable>
      <Text color="fg.muted" fontSize="sm">
        {t(`${KEY}.warning`, { count: report.inUseCount })}
        {hasQueue ? ` ${t(`${KEY}.queueWarning`, { count: report.inUseCount })}` : ''}
      </Text>
      {unchecked}
    </Stack>
  );
};

/**
 * The settled half of the usage check, loaded once a dialog shows one: the report, or a warning that the check failed.
 * `GalleryMediaUsagePanel` owns the live region and the checking state around it.
 */
export const GalleryMediaUsageOutcome = ({
  check,
  isRechecking = false,
  isRetryDisabled = false,
  onRetry,
}: {
  check: GalleryMediaUsageSettledCheck;
  /** A retry runs: the answer stays, the checking state in place of its warning. */
  isRechecking?: boolean;
  /** Retry stays in place but cannot run, such as while the deletion it would inform is already under way. */
  isRetryDisabled?: boolean;
  onRetry?: () => void;
}) => {
  const { t } = useTranslation();

  if (check.status === 'ready') {
    return (
      <UsageReport
        isRechecking={isRechecking}
        isRetryDisabled={isRetryDisabled}
        report={check.report}
        onRetry={onRetry}
      />
    );
  }

  return isRechecking ? (
    <GalleryMediaUsageChecking />
  ) : (
    <UsageWarning isRetryDisabled={isRetryDisabled} message={t(`${KEY}.failed`)} onRetry={onRetry} />
  );
};
