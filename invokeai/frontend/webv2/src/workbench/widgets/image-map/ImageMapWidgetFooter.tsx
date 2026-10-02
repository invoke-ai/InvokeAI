import type { WidgetViewProps } from '@workbench/widgetContracts';

import { HStack, Icon, Text } from '@chakra-ui/react';
import { IconButton, Tooltip } from '@platform/ui';
import { describeClusters, summarizeClusters } from '@workbench/image-map/clusterStats';
import { imageMapStore, refreshImageIndexStatus, refreshImageMapPoints } from '@workbench/image-map/imageMapStore';
import { isIndexing } from '@workbench/image-map/indexProgress';
import { RefreshCwIcon } from 'lucide-react';
import { useMemo } from 'react';

import { ImageIndexProgressInline } from './ImageIndexProgress';

/** Stable identity for the pre-`ready` renders, so the memo below never re-runs on null. */
const NO_POINTS: never[] = [];

const handleRefresh = () => {
  void refreshImageMapPoints();
  // Refresh counts too, recovering final status events missed while offline.
  refreshImageIndexStatus();
};

export const ImageMapWidgetFooter = (_props: WidgetViewProps) => {
  const { data, indexCounts, indexUpdatedAt, loadState } = imageMapStore.useSnapshot();
  // Before the early return, and keyed on the array the store swaps wholesale.
  // The footer re-renders on every index-progress tick, and those patch other
  // snapshot keys and leave `data` identity intact — so the count runs once per
  // points refresh rather than once per tick, which matters at 170k points.
  const points = data?.points ?? NO_POINTS;
  const clusterSummary = useMemo(() => describeClusters(summarizeClusters(points)), [points]);

  // Show footer controls only with a rendered map; other data states own their explanations and retry controls.
  if (loadState === 'idle' || !data || data.state !== 'ready') {
    return null;
  }

  const indexing = isIndexing(indexCounts);
  // Explain failed images when indexing drains below total.
  const skipped = indexCounts && indexCounts.pending === 0 && indexCounts.failed > 0;

  return (
    <HStack borderTopWidth="1px" color="fg.muted" fontSize="2xs" gap="2" justify="space-between" px="3" py="1" w="full">
      <HStack gap="2" minW="0">
        <Text whiteSpace="nowrap">{data.pointCount.toLocaleString()} points</Text>
        {/* Every sibling in this row is nowrap, so this is the only segment
            flex can shrink and it absorbs all of the row's overflow: below
            roughly 500px the numbers are clipped off the page and the tooltip
            is the only way left to read them. */}
        <Tooltip content={clusterSummary}>
          <Text truncate>· {clusterSummary}</Text>
        </Tooltip>
        {data.stale ? <Text whiteSpace="nowrap">· updating…</Text> : null}
        {indexing ? (
          <>
            <Text>·</Text>
            <ImageIndexProgressInline counts={indexCounts} updatedAt={indexUpdatedAt} />
          </>
        ) : null}
        {skipped ? (
          <Tooltip content="These items repeatedly failed to embed and were given up on.">
            <Text truncate>· {indexCounts.failed} skipped</Text>
          </Tooltip>
        ) : null}
      </HStack>
      <Tooltip content="Refresh map">
        <IconButton aria-label="Refresh map" color="fg.muted" size="2xs" variant="ghost" onClick={handleRefresh}>
          <Icon as={RefreshCwIcon} boxSize="3" />
        </IconButton>
      </Tooltip>
    </HStack>
  );
};
