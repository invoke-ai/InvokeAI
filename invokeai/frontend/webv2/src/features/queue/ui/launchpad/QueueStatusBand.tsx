import { Flex, Icon, Text } from '@chakra-ui/react';
import { queueBackend } from '@features/queue/data/httpRealtimeQueueBackend';
import { queueReadModelOptions } from '@features/queue/data/queries';
import { useQuery } from '@tanstack/react-query';
import { LoaderCircleIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

/**
 * What the backend is doing right now, on the home screen.
 *
 * The Launchpad used to surface no application state at all — the home screen
 * of an image generator said nothing about whether anything was generating.
 * This is deliberately a single line: the Queue widget in the editor is where
 * the detail lives.
 *
 * Scoped to every job rather than the active project, because Home is not
 * inside a project.
 *
 * Reads `data/queries` directly rather than the feature's `publicApi`: that
 * module constructs the realtime runtime and coordinator at import time, so
 * going through it would pull the whole Queue runtime onto the home screen for
 * two numbers.
 */

const ALL_JOBS_SCOPE = {} as const;

/**
 * Socket-driven invalidation lives in the editor's Queue widget runtime, and
 * the query client has window-focus refetching turned off, so without this the
 * band would show whatever the counts were when Home first mounted and never
 * move again.
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
      <Icon aria-hidden as={LoaderCircleIcon} boxSize="3.5" color="fg.muted" />
      <Text fontSize="xs" fontWeight="600">
        {t('launchpad.home.queue.summary', { inProgress, pending })}
      </Text>
    </Flex>
  );
};
