import type { ImageIndexCounts } from '@workbench/image-map/indexProgress';

import { Button, HStack, Progress, Stack, Text } from '@chakra-ui/react';
import { Tooltip } from '@platform/ui';
import { describeIndexProgress } from '@workbench/image-map/indexProgress';
import { useEffect, useState } from 'react';

const PROGRESS_LABEL = 'Image indexing progress';

/** Update age coarsely for minute-scale stall messages shared by panel and footer. */
const TICK_MS = 5_000;

/** How long the counts have been standing, re-read on a timer. */
const useCountsAge = (updatedAt: number | null): number => {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (updatedAt === null) {
      return;
    }

    const timer = setInterval(() => setNow(Date.now()), TICK_MS);

    return () => clearInterval(timer);
  }, [updatedAt]);

  // Clamped rather than reset on a new report: until the next tick `now` still
  // predates it, and a negative age would read as the future.
  return updatedAt === null ? 0 : Math.max(0, now - updatedAt);
};

interface ImageIndexProgressProps {
  counts: ImageIndexCounts;
  /** When the index last moved, or null if it never has been reported. */
  updatedAt: number | null;
}

/**
 * Show server-wide embedding progress only to admins receiving status events; non-admins see the empty state
 * instead of a frozen bar.
 */
export const ImageIndexProgressPanel = ({
  counts,
  error,
  onRetry,
  updatedAt,
}: ImageIndexProgressProps & { error?: string | null; onRetry: () => void }) => {
  const progress = describeIndexProgress(counts, useCountsAge(updatedAt));

  return (
    <Stack align="center" gap="2" maxW="sm" textAlign="center" w="full">
      <Text fontWeight="semibold">Indexing gallery</Text>
      <Text color="fg.muted" fontSize="sm">
        Images and videos are being embedded so they can be mapped. The map appears here on its own once enough of them
        are done — you can keep working in the meantime.
      </Text>
      <Stack gap="1" mt="2" w="full">
        <Progress.Root max={100} size="sm" value={progress.percent}>
          {/* The name goes on the track: that is the element carrying
              role="progressbar", and Chakra otherwise names it "25%", which
              tells a screen reader the number but never what it counts. */}
          <Progress.Track aria-label={PROGRESS_LABEL} aria-valuenow={progress.percent}>
            <Progress.Range />
          </Progress.Track>
        </Progress.Root>
        <HStack color="fg.muted" fontSize="xs" justify="space-between">
          <Text fontVariantNumeric="tabular-nums">{progress.counts}</Text>
          <Text fontVariantNumeric="tabular-nums">{progress.percent}%</Text>
        </HStack>
        {/* States the fact and stops there. Nothing here can tell an indexer
            waiting out a generation from one that has died, and the first is
            routine, so neither may be claimed. */}
        {progress.stale ? (
          <Text color="fg.subtle" fontSize="xs">
            {progress.stale}
          </Text>
        ) : null}
        {progress.skipped ? (
          <Text color="fg.subtle" fontSize="xs">
            {progress.skipped}
          </Text>
        ) : null}
      </Stack>
      {error ? (
        // Wrap server URLs/identifiers anywhere to prevent min-content overflow in narrow panels.
        <Text color="fg.error" fontSize="xs" maxW="full" minW="0" mt="2" overflowWrap="anywhere" role="alert">
          {error}
        </Text>
      ) : null}
      {/* Refresh is the only control before map readiness and recovers counts missed while offline. */}
      <Button mt={error ? '0' : '2'} onClick={onRetry} size="xs" variant="outline">
        {error ? 'Retry' : 'Check again'}
      </Button>
    </Stack>
  );
};

/**
 * Overlay rebuild progress while the stale map remains usable; explain missing labels during vocabulary embedding
 * rebuilds.
 */
export const ImageIndexActivityBadge = ({ counts, updatedAt }: ImageIndexProgressProps) => {
  const progress = describeIndexProgress(counts, useCountsAge(updatedAt));
  const label = progress.stale
    ? `Indexing ${progress.counts} · ${progress.stale}`
    : `Indexing ${progress.counts}. The map and its cluster labels update as images finish.`;

  return (
    // Positioned by the map's overlay column, which passes pointer events
    // through to the plot; the badge takes its own back so the tooltip opens.
    <Tooltip content={label}>
      <HStack
        bg="bg.subtle"
        borderColor="border.subtle"
        borderRadius="md"
        borderWidth="1px"
        color="fg.muted"
        fontSize="2xs"
        gap="1.5"
        maxW="full"
        minW="0"
        pointerEvents="auto"
        px="2"
        py="1"
        title={label}
      >
        <Progress.Root flexShrink="0" max={100} size="xs" value={progress.percent} w="10">
          <Progress.Track aria-label={`${PROGRESS_LABEL}: ${label}`} aria-valuenow={progress.percent}>
            <Progress.Range />
          </Progress.Track>
        </Progress.Root>
        <Text fontVariantNumeric="tabular-nums" truncate>
          indexing {progress.compact}
        </Text>
      </HStack>
    </Tooltip>
  );
};

export const ImageIndexProgressInline = ({ counts, updatedAt }: ImageIndexProgressProps) => {
  const progress = describeIndexProgress(counts, useCountsAge(updatedAt));
  const label = progress.stale ? `Indexing ${progress.counts} · ${progress.stale}` : `Indexing ${progress.counts}`;

  return (
    <Tooltip content={label}>
      {/* Use minW=0 and truncation so six-digit footer counts cannot cover refresh in narrow panels. */}
      <HStack gap="1.5" minW="0" overflow="hidden" title={label}>
        <Progress.Root flexShrink="0" max={100} size="xs" value={progress.percent} w="10">
          {/* The counts go in the name too: at the widget's minimum width the
              label beside it truncates to a couple of characters, and the
              tooltip carrying the full text is hover-only. */}
          <Progress.Track aria-label={`${PROGRESS_LABEL}: ${label}`} aria-valuenow={progress.percent}>
            <Progress.Range />
          </Progress.Track>
        </Progress.Root>
        <Text fontVariantNumeric="tabular-nums" truncate>
          indexing {progress.compact}
        </Text>
      </HStack>
    </Tooltip>
  );
};
