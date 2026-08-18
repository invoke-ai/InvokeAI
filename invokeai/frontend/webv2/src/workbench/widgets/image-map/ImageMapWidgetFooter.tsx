import type { WidgetViewProps } from '@workbench/widgetContracts';

import { HStack, Icon, Text } from '@chakra-ui/react';
import { IconButton, Tooltip } from '@platform/ui';
import { imageMapStore, refreshImageMapPoints } from '@workbench/image-map/imageMapStore';
import { isIndexing } from '@workbench/image-map/indexProgress';
import { RefreshCwIcon } from 'lucide-react';

import { ImageIndexProgressInline } from './ImageIndexProgress';

const handleRefresh = () => {
  void refreshImageMapPoints();
};

/**
 * Widget-chrome footer: point count, a stale hint while a recompute is
 * pending, embedding-index progress (admins only receive those events), and
 * a manual refresh.
 */
export const ImageMapWidgetFooter = (_props: WidgetViewProps) => {
  const { data, indexCounts, indexRate, loadState } = imageMapStore.useSnapshot();

  // Only the states that actually show a map. `disabled`, `model_missing` and
  // `empty` all carry data too, and each renders its own explanation — a
  // "0 points" line with a Refresh button under "Image indexing is off" says
  // nothing, and `computing` already offers its own "Check again".
  if (loadState === 'idle' || !data || data.state !== 'ready') {
    return null;
  }

  const indexing = isIndexing(indexCounts);
  // Once the queue drains, images given up on are the only reason the index
  // can settle short of `total` — without saying so the count simply stops
  // below the total with nothing to explain it.
  const skipped = indexCounts && indexCounts.pending === 0 && indexCounts.failed > 0;

  return (
    <HStack borderTopWidth="1px" color="fg.muted" fontSize="2xs" gap="2" justify="space-between" px="3" py="1" w="full">
      <HStack gap="2" minW="0">
        <Text whiteSpace="nowrap">{data.pointCount} points</Text>
        {data.stale ? <Text whiteSpace="nowrap">· updating…</Text> : null}
        {indexing ? (
          <>
            <Text>·</Text>
            <ImageIndexProgressInline counts={indexCounts} rate={indexRate} />
          </>
        ) : null}
        {skipped ? (
          <Tooltip content="These images repeatedly failed to embed and were given up on.">
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
