import { Flex } from '@invoke-ai/ui-library';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { InpaintMask } from 'features/controlLayers/components/InpaintMask/InpaintMask';
import { IPAdapterList } from 'features/controlLayers/components/IPAdapter/IPAdapterList';
import { LayerEntityList } from 'features/controlLayers/components/Layer/LayerEntityList';
import { RegionalGuidanceEntityList } from 'features/controlLayers/components/RegionalGuidance/RegionalGuidanceEntityList';
import { memo } from 'react';

export const CanvasEntityList = memo(() => {
  return (
    <ScrollableContent>
      <Flex flexDir="column" gap={2} data-testid="control-layers-layer-list">
        <InpaintMask />
        <RegionalGuidanceEntityList />
        <IPAdapterList />
        <LayerEntityList />
      </Flex>
    </ScrollableContent>
  );
});

CanvasEntityList.displayName = 'CanvasEntityList';
