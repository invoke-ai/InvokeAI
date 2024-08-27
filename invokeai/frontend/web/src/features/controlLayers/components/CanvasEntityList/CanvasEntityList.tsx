import { Flex } from '@invoke-ai/ui-library';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { CanvasEntityOpacity } from 'features/controlLayers/components/common/CanvasEntityOpacity';
import { ControlLayerEntityList } from 'features/controlLayers/components/ControlLayer/ControlLayerEntityList';
import { InpaintMaskList } from 'features/controlLayers/components/InpaintMask/InpaintMaskList';
import { IPAdapterList } from 'features/controlLayers/components/IPAdapter/IPAdapterList';
import { RasterLayerEntityList } from 'features/controlLayers/components/RasterLayer/RasterLayerEntityList';
import { RegionalGuidanceEntityList } from 'features/controlLayers/components/RegionalGuidance/RegionalGuidanceEntityList';
import { memo } from 'react';

export const CanvasEntityList = memo(() => {
  return (
    <ScrollableContent>
      <Flex flexDir="column" gap={2} pt={2} data-testid="control-layers-layer-list" w="full" h="full">
        <CanvasEntityOpacity />
        <InpaintMaskList />
        <RegionalGuidanceEntityList />
        <IPAdapterList />
        <ControlLayerEntityList />
        <RasterLayerEntityList />
      </Flex>
    </ScrollableContent>
  );
});

CanvasEntityList.displayName = 'CanvasEntityList';
