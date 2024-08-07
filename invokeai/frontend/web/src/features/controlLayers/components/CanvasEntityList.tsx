/* eslint-disable i18next/no-literal-string */
import { Flex } from '@invoke-ai/ui-library';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { ControlAdapterList } from 'features/controlLayers/components/ControlAdapter/ControlAdapterList';
import { IM } from 'features/controlLayers/components/InpaintMask/IM';
import { IPAEntityList } from 'features/controlLayers/components/IPAdapter/IPAEntityList';
import { LayerEntityList } from 'features/controlLayers/components/Layer/LayerEntityList';
import { RGEntityList } from 'features/controlLayers/components/RegionalGuidance/RGEntityList';
import { memo } from 'react';

export const CanvasEntityList = memo(() => {
  return (
    <ScrollableContent>
      <Flex flexDir="column" gap={2} data-testid="control-layers-layer-list">
        <IM />
        <RGEntityList />
        <ControlAdapterList />
        <IPAEntityList />
        <LayerEntityList />
      </Flex>
    </ScrollableContent>
  );
});

CanvasEntityList.displayName = 'CanvasEntityList';
