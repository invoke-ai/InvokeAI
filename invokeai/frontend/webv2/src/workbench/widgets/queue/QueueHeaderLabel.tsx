import { Stack, Text } from '@chakra-ui/react';

import { useScopedQueueCounts } from './queueScope';

/**
 * The Queue widget's header title: "Queue" over a live "N generating · M waiting"
 * summary. Rendered as the manifest `label`, so it sits in the standard frame
 * header. Counts come straight from the server-wide status, with tabular
 * numerals so the summary doesn't jitter as items move through the queue.
 */
export const QueueHeaderLabel = () => {
  const counts = useScopedQueueCounts();
  const isGenerating = counts.in_progress > 0;

  return (
    <Stack gap="0.5" minW="0">
      <Text fontSize="xs" fontWeight="700" lineHeight="1.15">
        Queue
      </Text>
      <Text color="fg.subtle" fontSize="2xs" fontVariantNumeric="tabular-nums" lineHeight="1.15" truncate mb="-1.5">
        <Text as="span" color={isGenerating ? 'accent.solid' : 'fg.subtle'} fontWeight={isGenerating ? '600' : '400'}>
          {counts.in_progress} generating
        </Text>{' '}
        · {counts.pending} waiting
      </Text>
    </Stack>
  );
};
