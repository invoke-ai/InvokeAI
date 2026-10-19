import { Collapsible, HStack, Icon, Stack, Text } from '@chakra-ui/react';
import { invocationCacheStatusOptions, runInvocationCacheCommand } from '@features/queue/data/invocationCacheQueries';
import { queryClient } from '@platform/query/client';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button } from '@platform/ui/Button';
import { useQuery } from '@tanstack/react-query';
import { ChevronUpIcon, LayersIcon, PauseIcon, PlayIcon, Trash2Icon } from 'lucide-react';
import { useCallback, useId, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { useQueueUi } from './QueueUiContext';

const INDICATOR_OPEN = { transform: 'rotate(180deg)' } as const;
const TRIGGER_HOVER = { color: 'fg' } as const;

const CacheStat = ({ label, value }: { label: string; value: number }) => (
  <Text color="fg.muted" fontSize="xs">
    {label}{' '}
    <Text as="span" color="fg" fontVariantNumeric="tabular-nums" fontWeight="600">
      {value}
    </Text>
  </Text>
);

/**
 * The invocation cache keeps node outputs so an unchanged node is not re-executed; it is not the model cache. The
 * server exposes it to administrators only, so the footer renders for them alone.
 */
export const InvocationCacheFooter = () => {
  const { t } = useTranslation();
  const { notify } = useQueueUi();
  const [expanded, setExpanded] = useState(false);
  const [busy, setBusy] = useState<'clear' | 'disable' | 'enable' | null>(null);
  const query = useQuery(invocationCacheStatusOptions());
  const status = query.data ?? null;

  const handleOpenChange = useCallback((event: { open: boolean }) => setExpanded(event.open), []);
  const run = useCallback(
    async (command: 'clear' | 'disable' | 'enable') => {
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
  const canCommand = status !== null && !hasNoCapacity && busy === null;

  const summary = query.isError
    ? t('widgets.queue.invocationCache.unavailable')
    : status
      ? hasNoCapacity
        ? t('widgets.queue.invocationCache.off')
        : status.enabled
          ? t('widgets.queue.invocationCache.sizeOfMax', { max: status.maxSize, size: status.size })
          : t('widgets.queue.invocationCache.disabled')
      : t('widgets.queue.invocationCache.loading');

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
          {query.isError ? (
            <Text color="fg.error" fontSize="xs">
              {getApiErrorMessage(query.error, t('widgets.queue.invocationCache.unavailable'))}
            </Text>
          ) : (
            <HStack gap="3">
              <CacheStat label={t('widgets.queue.invocationCache.hits')} value={status?.hits ?? 0} />
              <CacheStat label={t('widgets.queue.invocationCache.misses')} value={status?.misses ?? 0} />
              <CacheStat label={t('widgets.queue.invocationCache.cached')} value={status?.size ?? 0} />
            </HStack>
          )}
          {hasNoCapacity ? (
            <Text color="fg.muted" fontSize="xs" id={noCapacityId}>
              {t('widgets.queue.invocationCache.noCapacity')}
            </Text>
          ) : null}
          <HStack>
            <Button
              aria-describedby={commandDescription}
              disabled={!canCommand}
              loading={busy === 'clear'}
              size="sm"
              variant="surface"
              onClick={onClear}
            >
              <Icon as={Trash2Icon} boxSize="3" />
              {t('widgets.queue.invocationCache.clear')}
            </Button>
            <Button
              aria-describedby={commandDescription}
              disabled={!canCommand}
              loading={busy === 'enable' || busy === 'disable'}
              size="sm"
              variant="surface"
              onClick={onToggleEnabled}
            >
              <Icon as={status?.enabled ? PauseIcon : PlayIcon} boxSize="3" />
              {status?.enabled ? t('widgets.queue.invocationCache.disable') : t('widgets.queue.invocationCache.enable')}
            </Button>
          </HStack>
        </Stack>
      </Collapsible.Content>

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
        <Icon as={LayersIcon} boxSize="3.5" color="fg.subtle" flexShrink={0} />
        <Text color="fg.muted" flex="1" fontSize="xs" fontWeight="700" letterSpacing="0.06em" textTransform="uppercase">
          {t('widgets.queue.invocationCache.label')}
        </Text>
        <Text color="fg.muted" fontSize="xs" fontVariantNumeric="tabular-nums">
          {summary}
        </Text>
        <Collapsible.Indicator _open={INDICATOR_OPEN} transition="transform var(--wb-motion-duration-slow)">
          <Icon as={ChevronUpIcon} boxSize="4" color="fg.subtle" flexShrink={0} />
        </Collapsible.Indicator>
      </Collapsible.Trigger>
    </Collapsible.Root>
  );
};
