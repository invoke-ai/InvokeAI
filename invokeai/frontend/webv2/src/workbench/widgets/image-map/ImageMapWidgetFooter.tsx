import type { WidgetViewProps } from '@workbench/widgetContracts';

import { HStack, Icon, Text } from '@chakra-ui/react';
import { IconButton, Tooltip } from '@platform/ui';
import { imageMapStore, refreshImageMapPoints } from '@workbench/image-map/imageMapStore';
import { RefreshCwIcon } from 'lucide-react';

const handleRefresh = () => {
  void refreshImageMapPoints();
};

/**
 * Widget-chrome footer: point count, a stale hint while a recompute is
 * pending, embedding-index progress (admins only receive those events), and
 * a manual refresh.
 */
export const ImageMapWidgetFooter = (_props: WidgetViewProps) => {
  const { data, indexCounts, loadState } = imageMapStore.useSnapshot();

  if (loadState === 'idle' || !data) {
    return null;
  }

  const indexing = indexCounts && indexCounts.pending > 0;

  return (
    <HStack borderTopWidth="1px" color="fg.muted" fontSize="2xs" gap="2" justify="space-between" px="3" py="1" w="full">
      <HStack gap="2" minW="0">
        <Text whiteSpace="nowrap">{data.pointCount} points</Text>
        {data.stale ? <Text whiteSpace="nowrap">· updating…</Text> : null}
        {indexing ? (
          <Text truncate>
            · indexing {indexCounts.embedded}/{indexCounts.total}
          </Text>
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
