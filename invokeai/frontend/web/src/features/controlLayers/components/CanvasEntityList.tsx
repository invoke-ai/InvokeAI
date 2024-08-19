import { Flex } from '@invoke-ai/ui-library';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { CanvasEntityOpacity } from 'features/controlLayers/components/common/CanvasEntityOpacity';
import { ControlLayerEntityList } from 'features/controlLayers/components/ControlLayer/ControlLayerEntityList';
import { InpaintMask } from 'features/controlLayers/components/InpaintMask/InpaintMask';
import { IPAdapterList } from 'features/controlLayers/components/IPAdapter/IPAdapterList';
import { RasterLayerEntityList } from 'features/controlLayers/components/RasterLayer/RasterLayerEntityList';
import { RegionalGuidanceEntityList } from 'features/controlLayers/components/RegionalGuidance/RegionalGuidanceEntityList';
import { memo } from 'react';

export const CanvasEntityList = memo(() => {
  return (
    <ScrollableContent>
      <Flex flexDir="column" gap={4} pt={2} data-testid="control-layers-layer-list">
        <CanvasEntityOpacity />
        <InpaintMask />
        <RegionalGuidanceEntityList />
        <IPAdapterList />
        <ControlLayerEntityList />
        <RasterLayerEntityList />
      </Flex>
    </ScrollableContent>
  );
});

CanvasEntityList.displayName = 'CanvasEntityList';
