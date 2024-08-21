import { Flex } from '@invoke-ai/ui-library';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { CanvasEntityOpacity } from 'features/controlLayers/components/common/CanvasEntityOpacity';
import { ControlLayerEntityList } from 'features/controlLayers/components/ControlLayer/ControlLayerEntityList';
import { InpaintMaskList } from 'features/controlLayers/components/InpaintMask/InpaintMaskList';
import { IPAdapterList } from 'features/controlLayers/components/IPAdapter/IPAdapterList';
import { RasterLayerEntityList } from 'features/controlLayers/components/RasterLayer/RasterLayerEntityList';
import { RegionalGuidanceEntityList } from 'features/controlLayers/components/RegionalGuidance/RegionalGuidanceEntityList';
import { CanvasManagerProviderGate } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { memo } from 'react';

export const CanvasEntityList = memo(() => {
  return (
    <CanvasManagerProviderGate>
      <ScrollableContent>
        <Flex flexDir="column" gap={4} pt={2} data-testid="control-layers-layer-list">
          <CanvasEntityOpacity />
          <InpaintMaskList />
          <RegionalGuidanceEntityList />
          <IPAdapterList />
          <ControlLayerEntityList />
          <RasterLayerEntityList />
        </Flex>
      </ScrollableContent>
    </CanvasManagerProviderGate>
  );
});

CanvasEntityList.displayName = 'CanvasEntityList';
