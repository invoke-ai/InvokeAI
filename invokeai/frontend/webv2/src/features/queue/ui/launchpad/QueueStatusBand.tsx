import { Flex, Icon, Spinner, Text } from '@chakra-ui/react';
import { queueBackend } from '@features/queue/data/httpRealtimeQueueBackend';
import { queueReadModelOptions } from '@features/queue/data/queries';
import { useQuery } from '@tanstack/react-query';
import { HourglassIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

/**
 * Home shows server-wide activity outside project scope. Read queries directly to avoid loading Queue runtime for
 * summary counts.
 */

const ALL_JOBS_SCOPE = {} as const;

/**
 * Refresh Home counts here because editor realtime invalidation is unmounted and query focus refetching is
 * disabled.
 */
const QUEUE_POLL_INTERVAL_MS = 5_000;

export const QueueStatusBand = () => {
  const { t } = useTranslation();
  const { data } = useQuery({
    ...queueReadModelOptions(queueBackend, ALL_JOBS_SCOPE),
    refetchInterval: QUEUE_POLL_INTERVAL_MS,
    refetchIntervalInBackground: false,
  });

  const counts = data?.status.queue;
  const inProgress = counts?.inProgress ?? 0;
  const pending = counts?.pending ?? 0;

  if (inProgress === 0 && pending === 0) {
    return null;
  }

  return (
    <Flex
      align="center"
      aria-live="polite"
      bg="bg.subtle"
      borderColor="border.subtle"
      borderWidth="1px"
      gap="2"
      px="3"
      py="2"
      rounded="lg"
    >
      {/* Animate only running work; waiting-only queues use a static glyph. */}
      {inProgress > 0 ? (
        <Spinner color="fg.muted" size="xs" />
      ) : (
        <Icon aria-hidden as={HourglassIcon} boxSize="3.5" color="fg.muted" />
      )}
      <Text fontSize="xs" fontWeight="600">
        {t('launchpad.home.queue.summary', { inProgress, pending })}
      </Text>
    </Flex>
  );
};
