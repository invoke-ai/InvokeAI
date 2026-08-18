import type { ImageIndexCounts } from '@workbench/image-map/indexProgress';

import { HStack, Progress, Stack, Text } from '@chakra-ui/react';
import { Tooltip } from '@platform/ui';
import { describeIndexProgress } from '@workbench/image-map/indexProgress';

const PROGRESS_LABEL = 'Image indexing progress';

interface ImageIndexProgressProps {
  counts: ImageIndexCounts;
  /** Images per second, or null while it is still being measured. */
  rate: number | null;
}

/**
 * Backfill progress for the embedding index, shown in place of the map while
 * there is nothing to draw yet. The counts are pushed by `image_index_status`
 * (admins only, which in single-user mode is everyone), so a non-admin sees
 * the plain "nothing to map yet" message instead of a bar that never moves.
 */
export const ImageIndexProgressPanel = ({ counts, rate }: ImageIndexProgressProps) => {
  const progress = describeIndexProgress(counts, rate);

  return (
    <Stack align="center" gap="2" maxW="sm" textAlign="center" w="full">
      <Text fontWeight="semibold">Indexing your gallery</Text>
      <Text color="fg.muted" fontSize="sm">
        Your images are being embedded so they can be mapped. The map appears here on its own once enough of them are
        done — you can keep working in the meantime.
      </Text>
      <Stack gap="1" mt="2" w="full">
        <Progress.Root max={100} size="sm" value={progress.percent}>
          {/* The name goes on the track: that is the element carrying
              role="progressbar", and Chakra otherwise names it "25%", which
              tells a screen reader the number but never what it counts. */}
          <Progress.Track aria-label={PROGRESS_LABEL}>
            <Progress.Range />
          </Progress.Track>
        </Progress.Root>
        <HStack color="fg.muted" fontSize="xs" justify="space-between">
          <Text fontVariantNumeric="tabular-nums">{progress.counts}</Text>
          <Text fontVariantNumeric="tabular-nums">{Math.round(progress.percent)}%</Text>
        </HStack>
        <Text color="fg.subtle" fontSize="xs">
          {progress.eta}
        </Text>
        {progress.skipped ? (
          <Text color="fg.subtle" fontSize="xs">
            {progress.skipped}
          </Text>
        ) : null}
      </Stack>
    </Stack>
  );
};

/**
 * The same progress in one line, for the widget footer once the map itself is
 * on screen: the bar is the glanceable part, the time estimate is one hover
 * away rather than competing with the point count for the width.
 */
export const ImageIndexProgressInline = ({ counts, rate }: ImageIndexProgressProps) => {
  const progress = describeIndexProgress(counts, rate);

  return (
    <Tooltip content={`Indexing ${progress.counts} · ${progress.eta}`}>
      <HStack gap="1.5" minW="0">
        <Progress.Root max={100} minW="10" size="xs" value={progress.percent} w="10">
          <Progress.Track aria-label={PROGRESS_LABEL}>
            <Progress.Range />
          </Progress.Track>
        </Progress.Root>
        <Text fontVariantNumeric="tabular-nums" whiteSpace="nowrap">
          indexing {counts.embedded}/{counts.total}
        </Text>
      </HStack>
    </Tooltip>
  );
};
