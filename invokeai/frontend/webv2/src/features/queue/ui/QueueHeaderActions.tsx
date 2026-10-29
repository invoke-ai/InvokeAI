import {
  Grid,
  HStack,
  Icon,
  Popover,
  Portal,
  Progress,
  Separator,
  Stack,
  Text,
  VisuallyHidden,
} from '@chakra-ui/react';
import { invocationCacheStatusOptions, runInvocationCacheCommand } from '@features/queue/data/invocationCacheQueries';
import { formatCount } from '@platform/i18n/languages';
import { queryClient } from '@platform/query/client';
import { useMountEffect } from '@platform/react/useMountEffect';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, IconButton } from '@platform/ui/Button';
import { PopoverContent } from '@platform/ui/Popover';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { useQuery } from '@tanstack/react-query';
import { DatabaseIcon } from 'lucide-react';
import { useCallback, useId, useMemo, useRef, useState, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

import {
  clearModelCache,
  formatModelCacheBytes,
  formatModelCacheRange,
  getModelCacheClearToast,
  getModelCacheUsage,
  watchModelCacheStats,
  useModelCacheStats,
} from './modelCacheStore';
import { useQueueUi } from './QueueUiContext';

const POSITIONING = { placement: 'bottom-end' } as const;
/** Label, bar, value, hit rate, Clear: one grid, so the columns line up across rows. */
const ROW_COLUMNS = 'auto minmax(3rem, 1fr) auto 2.25rem auto';
const FULL_ROW = '1 / -1';
const VALUE_COLUMNS = '2 / 5';

/** The queue header opens both server caches in one popover; the invocation cache row is for administrators only. */
export const QueueHeaderActions = () => {
  const { t } = useTranslation();
  const triggerIds = useTooltipTriggerIds();
  const titleId = useId();
  // Zag names the dialog by its title only when the title exists at mount; lazy content mounts later, so name it here.
  const ids = useMemo(() => ({ ...triggerIds, title: titleId }), [titleId, triggerIds]);
  const contentRef = useRef<HTMLDivElement>(null);
  // Focus lands on the panel, not on its first control: that is a Clear button.
  const getInitialFocus = useCallback(() => contentRef.current, []);

  return (
    <Popover.Root ids={ids} initialFocusEl={getInitialFocus} lazyMount positioning={POSITIONING} unmountOnExit>
      <Tooltip content={t('widgets.queue.cache.title')} ids={triggerIds}>
        <Popover.Trigger asChild>
          <IconButton aria-label={t('widgets.queue.cache.title')} color="fg.muted" size="sm" variant="ghost">
            <Icon as={DatabaseIcon} boxSize="3.5" />
          </IconButton>
        </Popover.Trigger>
      </Tooltip>
      <Portal>
        <Popover.Positioner>
          <PopoverContent ref={contentRef} aria-labelledby={titleId} p="3" w="21rem">
            <CachePanel />
          </PopoverContent>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
  );
};

/** Mounted only while open, so opening refreshes both readings and a closed popover polls nothing. */
const CachePanel = () => {
  const { t } = useTranslation();
  const { canManageInvocationCache } = useQueueUi();

  return (
    <Stack gap="2.5">
      <Popover.Title fontSize="md" fontWeight="800" lineHeight="1rem">
        {t('widgets.queue.cache.title')}
      </Popover.Title>
      <Grid alignItems="center" columnGap="2.5" rowGap="2" templateColumns={ROW_COLUMNS}>
        {canManageInvocationCache ? (
          <InvocationCacheSection>
            <RowSeparator />
            <ModelCacheRow />
          </InvocationCacheSection>
        ) : (
          <ModelCacheRow />
        )}
      </Grid>
    </Stack>
  );
};

const RowSeparator = () => <Separator borderColor="border.subtle" gridColumn={FULL_ROW} />;

const hitRatePercent = (hits: number, misses: number): number | null =>
  hits + misses > 0 ? Math.round((hits / (hits + misses)) * 100) : null;

const HitRate = ({ hits, misses }: { hits: number; misses: number }) => {
  const { t } = useTranslation();
  const percent = hitRatePercent(hits, misses);

  return (
    <Text
      color={percent === null ? 'fg.subtle' : 'fg.muted'}
      fontSize="xs"
      fontVariantNumeric="tabular-nums"
      fontWeight="600"
      textAlign="end"
    >
      <span aria-hidden>{percent === null ? '—' : `${percent}%`}</span>
      <VisuallyHidden>
        {percent === null ? t('widgets.queue.cache.noLookups') : t('widgets.queue.cache.hitRate', { percent })}
      </VisuallyHidden>
    </Text>
  );
};

const RowLabel = ({ children }: { children: string }) => (
  <Text fontSize="sm" fontWeight="700" minW="4.5rem">
    {children}
  </Text>
);

interface UsageCellsProps {
  colorPalette: string;
  hits: number;
  /** The bar's accessible name, with the reading in words. */
  label: string;
  misses: number;
  ratio: number;
  value: string;
}

/** Bar, value and hit rate: three cells of the panel's shared columns, not a box of their own. */
const UsageCells = ({ colorPalette, hits, label, misses, ratio, value }: UsageCellsProps) => (
  <>
    <Progress.Root colorPalette={colorPalette} max={1} minW="0" value={ratio}>
      {/* The track is the progressbar; its default name is only the percentage. */}
      <Progress.Track aria-label={label} borderRadius="full" h="1">
        <Progress.Range borderRadius="full" />
      </Progress.Track>
    </Progress.Root>
    <Text color="fg" fontSize="xs" fontVariantNumeric="tabular-nums" textAlign="end" whiteSpace="nowrap">
      {value}
    </Text>
    <HitRate hits={hits} misses={misses} />
  </>
);

/** Takes the bar, value and hit rate cells when there is no reading to show. */
const RowMessage = ({ children, tone = 'muted' }: { children: string; tone?: 'error' | 'muted' }) => (
  <Text color={tone === 'error' ? 'fg.error' : 'fg.muted'} fontSize="xs" gridColumn={VALUE_COLUMNS} minW="0">
    {children}
  </Text>
);

type InvocationCacheCommand = 'clear' | 'disable' | 'enable';

/**
 * The invocation cache keeps node outputs so an unchanged node is not re-executed; it is not the model cache. Its row
 * and the Enable/Disable footer share one status read and one command at a time; `children` sit between them.
 */
const InvocationCacheSection = ({ children }: { children: ReactNode }) => {
  const { t } = useTranslation();
  const { notify } = useQueueUi();
  const [busy, setBusy] = useState<InvocationCacheCommand | null>(null);
  const query = useQuery({ ...invocationCacheStatusOptions(), refetchOnMount: 'always' });
  const status = query.data ?? null;

  const run = useCallback(
    async (command: InvocationCacheCommand) => {
      setBusy(command);

      try {
        if (await runInvocationCacheCommand(queryClient, command)) {
          notify.success(t(`widgets.queue.invocationCache.${command}Done`));
        }
      } catch (error) {
        notify.error(
          t(`widgets.queue.invocationCache.${command}Failed`),
          getApiErrorMessage(error, t('widgets.queue.invocationCache.couldNotChange'))
        );
      } finally {
        setBusy(null);
      }
    },
    [notify, t]
  );
  const onClear = useCallback(() => run('clear'), [run]);
  const onToggleEnabled = useCallback(() => run(status?.enabled ? 'disable' : 'enable'), [run, status?.enabled]);
  // With no capacity the server ignores every command, so none is offered; only its configuration can turn it on.
  const hasNoCapacity = status?.maxSize === 0;
  const noCapacityId = useId();
  const commandDescription = hasNoCapacity ? noCapacityId : undefined;
  const canCommand = !query.isError && status !== null && !hasNoCapacity && busy === null;

  const usageLabel = status
    ? t('widgets.queue.cache.invocationUsage', { max: formatCount(status.maxSize), size: formatCount(status.size) })
    : '';

  return (
    <>
      <RowLabel>{t('widgets.queue.cache.invocations')}</RowLabel>
      {query.isError ? (
        <RowMessage tone="error">
          {getApiErrorMessage(query.error, t('widgets.queue.invocationCache.unavailable'))}
        </RowMessage>
      ) : !status ? (
        <RowMessage>{t('widgets.queue.cache.loading')}</RowMessage>
      ) : hasNoCapacity ? (
        <RowMessage>{t('widgets.queue.invocationCache.off')}</RowMessage>
      ) : (
        <UsageCells
          // A disabled cache keeps its entries but reuses none, so its bar goes quiet.
          colorPalette={status.enabled ? 'pink' : 'gray'}
          hits={status.hits}
          label={usageLabel}
          misses={status.misses}
          ratio={Math.min(1, status.size / status.maxSize)}
          value={`${formatCount(status.size)}/${formatCount(status.maxSize)}`}
        />
      )}
      <Button
        aria-describedby={commandDescription}
        aria-label={t('widgets.queue.cache.clearInvocations')}
        disabled={!canCommand}
        loading={busy === 'clear'}
        size="xs"
        variant="ghost"
        onClick={onClear}
      >
        {t('widgets.queue.cache.clear')}
      </Button>
      {hasNoCapacity ? (
        <Text color="fg.muted" fontSize="xs" gridColumn={FULL_ROW} id={noCapacityId} mt="-1">
          {t('widgets.queue.invocationCache.noCapacity')}
        </Text>
      ) : null}
      {children}
      {/* Without a reading there is no state to switch from, so the footer waits for one. */}
      {status ? (
        <>
          <RowSeparator />
          <HStack gridColumn={FULL_ROW} justify="space-between" minH="6">
            <Text color="fg.muted" fontSize="xs">
              {query.isError
                ? null
                : status.enabled && !hasNoCapacity
                  ? t('widgets.queue.cache.invocationsOn')
                  : t('widgets.queue.cache.invocationsOff')}
            </Text>
            <Button
              aria-describedby={commandDescription}
              disabled={!canCommand}
              loading={busy === 'enable' || busy === 'disable'}
              size="xs"
              variant="ghost"
              onClick={onToggleEnabled}
            >
              {status.enabled ? t('widgets.queue.cache.disable') : t('widgets.queue.cache.enable')}
            </Button>
          </HStack>
        </>
      ) : null}
    </>
  );
};

/** Model-cache statistics and clearing; no backend endpoint disables the model cache. */
const ModelCacheRow = () => {
  const { t } = useTranslation();
  const { error, loadState, stats } = useModelCacheStats();
  const { notify } = useQueueUi();
  const [busy, setBusy] = useState(false);

  // Reads on open, then on each model load while open; nothing polls the stats while the popover is closed.
  useMountEffect(watchModelCacheStats);

  const { total, used } = getModelCacheUsage(stats);

  const handleClear = useCallback(async () => {
    setBusy(true);

    try {
      const clearResult = await clearModelCache();
      const toast = getModelCacheClearToast(clearResult, t);
      notify[toast.status](toast.title, toast.description);
    } catch (clearError) {
      notify.error(
        t('widgets.queue.modelCache.clearFailed'),
        getApiErrorMessage(clearError, t('widgets.queue.modelCache.couldNotClear'))
      );
    } finally {
      setBusy(false);
    }
  }, [notify, t]);

  return (
    <>
      <RowLabel>{t('widgets.queue.cache.models')}</RowLabel>
      {stats || (!error && loadState === 'loaded') ? (
        <UsageCells
          colorPalette="accent"
          hits={stats?.hits ?? 0}
          label={t('widgets.queue.cache.modelUsage', {
            total: formatModelCacheBytes(total),
            used: formatModelCacheBytes(used),
          })}
          misses={stats?.misses ?? 0}
          ratio={total > 0 ? Math.min(1, used / total) : 0}
          value={formatModelCacheRange(used, total)}
        />
      ) : error ? (
        <RowMessage tone="error">{error}</RowMessage>
      ) : (
        <RowMessage>{t('widgets.queue.cache.loading')}</RowMessage>
      )}
      <Button
        aria-label={t('widgets.queue.cache.clearModels')}
        loading={busy}
        size="xs"
        variant="ghost"
        onClick={handleClear}
      >
        {t('widgets.queue.cache.clear')}
      </Button>
    </>
  );
};
