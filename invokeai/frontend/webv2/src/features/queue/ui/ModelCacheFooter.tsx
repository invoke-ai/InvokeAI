import { Collapsible, HStack, Icon, Progress, Stack, Text } from '@chakra-ui/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button } from '@platform/ui/Button';
import { ChevronUpIcon, DatabaseIcon, Trash2Icon } from 'lucide-react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

import {
  clearModelCache,
  formatModelCacheBytes,
  getModelCacheClearToast,
  getModelCacheUsage,
  refreshModelCacheStats,
  useModelCacheStats,
} from './modelCacheStore';
import { useQueueUi } from './QueueUiContext';

const INDICATOR_OPEN = { transform: 'rotate(180deg)' } as const;
const TRIGGER_HOVER = { color: 'fg' } as const;

const CacheStat = ({ label, value }: { label: string; value: number }) => (
  <Text color="fg.subtle" fontSize="2xs">
    {label}{' '}
    <Text as="span" color="fg" fontVariantNumeric="tabular-nums" fontWeight="600">
      {value}
    </Text>
  </Text>
);

/**
 * Collapsible MODEL CACHE footer (manifest `footer` slot). Collapsed: a label, a
 * RAM-used / budget readout, and a fill bar. Expanded: hit/miss/loaded counts and
 * a Clear action. The cache has no enable/disable endpoint, so there is no
 * "Disable" toggle. Mirrors the model manager's `InstallQueueBar` collapse shape.
 */
export const ModelCacheFooter = () => {
  const { t } = useTranslation();
  const { stats } = useModelCacheStats();
  const { notify } = useQueueUi();
  const [expanded, setExpanded] = useState(false);
  const [busy, setBusy] = useState(false);

  useMountEffect(() => {
    void refreshModelCacheStats();
  });

  const { total, used } = getModelCacheUsage(stats);
  const ratio = total > 0 ? Math.min(1, used / total) : 0;

  const handleOpenChange = useCallback((event: { open: boolean }) => setExpanded(event.open), []);
  const handleClear = useCallback(async () => {
    setBusy(true);

    try {
      const clearResult = await clearModelCache();
      const toast = getModelCacheClearToast(clearResult, t);
      notify[toast.status](toast.title, toast.description);
    } catch (error) {
      notify.error(
        t('widgets.queue.modelCache.clearFailed'),
        getApiErrorMessage(error, t('widgets.queue.modelCache.couldNotClear'))
      );
    } finally {
      setBusy(false);
    }
  }, [notify, t]);

  return (
    <Collapsible.Root
      bg="bg.subtle"
      borderTopWidth={1}
      open={expanded}
      overflow="hidden"
      onOpenChange={handleOpenChange}
    >
      <Collapsible.Content>
        <Stack borderBottomWidth={1} gap="2" px="3" py="2.5">
          <HStack gap="3">
            <CacheStat label={t('widgets.queue.modelCache.hits')} value={stats?.hits ?? 0} />
            <CacheStat label={t('widgets.queue.modelCache.misses')} value={stats?.misses ?? 0} />
            <CacheStat label={t('widgets.queue.modelCache.loaded')} value={stats?.in_cache ?? 0} />
          </HStack>
          <HStack>
            <Button loading={busy} size="2xs" variant="surface" onClick={handleClear}>
              <Icon as={Trash2Icon} boxSize="3" />
              {t('widgets.queue.modelCache.clear')}
            </Button>
          </HStack>
        </Stack>
      </Collapsible.Content>

      <Stack gap="0">
        <Progress.Root
          aria-label={t('widgets.queue.modelCache.usage')}
          aria-valuetext={t('widgets.queue.modelCache.usedOfTotal', {
            total: formatModelCacheBytes(total),
            used: formatModelCacheBytes(used),
          })}
          colorPalette="accent"
          max={1}
          size="xs"
          value={ratio}
        >
          <Progress.Track bg="transparent">
            <Progress.Range />
          </Progress.Track>
        </Progress.Root>
        <Collapsible.Trigger
          alignItems="center"
          bg="transparent"
          color="inherit"
          display="flex"
          gap="2"
          px="3"
          py="2"
          textAlign="start"
          w="full"
          _hover={TRIGGER_HOVER}
        >
          <Icon as={DatabaseIcon} boxSize="3.5" color="fg.subtle" flexShrink={0} />
          <Text
            color="fg.muted"
            flex="1"
            fontSize="2xs"
            fontWeight="700"
            letterSpacing="0.06em"
            textTransform="uppercase"
          >
            {t('widgets.queue.modelCache.label')}
          </Text>
          <Text color="fg.subtle" fontSize="2xs" fontVariantNumeric="tabular-nums">
            {formatModelCacheBytes(used)} / {formatModelCacheBytes(total)}
          </Text>
          <Collapsible.Indicator _open={INDICATOR_OPEN} transition="transform var(--wb-motion-duration-slow)">
            <Icon as={ChevronUpIcon} boxSize="4" color="fg.subtle" flexShrink={0} />
          </Collapsible.Indicator>
        </Collapsible.Trigger>
      </Stack>
    </Collapsible.Root>
  );
};
