import type { ImageIndexCounts } from '@workbench/image-map/indexProgress';

import { Button, HStack, Progress, Stack, Text } from '@chakra-ui/react';
import { Tooltip } from '@platform/ui';
import { describeIndexProgress } from '@workbench/image-map/indexProgress';
import { useEffect, useState } from 'react';

const PROGRESS_LABEL = 'Image indexing progress';

/**
 * How often the age of the counts is re-read. The only thing it drives is a
 * minutes-scale "no progress" note, so this is deliberately coarse: the same
 * interval serves the panel and the footer, and neither announces a number
 * that moves every second.
 */
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
 * Backfill progress for the embedding index, shown in place of the map while
 * there is nothing to draw yet. The counts are pushed by `image_index_status`
 * (admins only, which in single-user mode is everyone), so a non-admin sees
 * the plain "nothing to map yet" message instead of a bar that never moves.
 *
 * The wording stays away from "your images": the backend's counts aggregate
 * every user's gallery, so for an admin on a multi-user server this is the
 * server's backlog and not necessarily any of their own.
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
      <Text fontWeight="semibold">Indexing images</Text>
      <Text color="fg.muted" fontSize="sm">
        Images are being embedded so they can be mapped. The map appears here on its own once enough of them are done —
        you can keep working in the meantime.
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
        // `overflowWrap: anywhere` because the message is whatever the server
        // said: a URL or a dotted identifier has no break opportunity, and a
        // centered flex item sizes to min-content, so it would otherwise run
        // off both edges of a widget clipped at 280px with no ellipsis.
        <Text color="fg.error" fontSize="xs" maxW="full" minW="0" mt="2" overflowWrap="anywhere" role="alert">
          {error}
        </Text>
      ) : null}
      {/* The widget's only control in this state — the footer renders nothing
          until the map itself is ready. It matters most when the counts are
          wrong: a client that was offline for the run's final report sits on a
          bar that will never move again, and this re-reads them. */}
      <Button mt={error ? '0' : '2'} onClick={onRetry} size="xs" variant="outline">
        {error ? 'Retry' : 'Check again'}
      </Button>
    </Stack>
  );
};

/**
 * The same progress in one line, for the widget footer once the map itself is
 * on screen: the bar is the glanceable part, the rest is one hover away rather
 * than competing with the point count for the width.
 */
export const ImageIndexProgressInline = ({ counts, updatedAt }: ImageIndexProgressProps) => {
  const progress = describeIndexProgress(counts, useCountsAge(updatedAt));
  const label = progress.stale ? `Indexing ${progress.counts} · ${progress.stale}` : `Indexing ${progress.counts}`;

  return (
    <Tooltip content={label}>
      {/* `minW="0"` and a truncating label, both of which the plain `<Text
          truncate>` this replaced got for free: the footer resizes down to
          280px and the counts can be six digits each, and without them the row
          grows past the panel edge and slides under the refresh button. */}
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
