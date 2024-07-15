/* eslint-disable i18next/no-literal-string */
import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { CAEntityList } from 'features/controlLayers/components/ControlAdapter/CAEntityList';
import { InitialImage } from 'features/controlLayers/components/InitialImage/InitialImage';
import { IM } from 'features/controlLayers/components/InpaintMask/IM';
import { IPAEntityList } from 'features/controlLayers/components/IPAdapter/IPAEntityList';
import { LayerEntityList } from 'features/controlLayers/components/Layer/LayerEntityList';
import { RGEntityList } from 'features/controlLayers/components/RegionalGuidance/RGEntityList';
import { memo } from 'react';

export const CanvasEntityList = memo(() => {
  const isCanvasSessionActive = useAppSelector((s) => s.canvasV2.session.isActive);

  return (
    <ScrollableContent>
      <Flex flexDir="column" gap={2} data-testid="control-layers-layer-list">
        {isCanvasSessionActive && <IM />}
        <RGEntityList />
        <CAEntityList />
        <IPAEntityList />
        <LayerEntityList />
        {!isCanvasSessionActive && <InitialImage />}
      </Flex>
    </ScrollableContent>
  );
});

CanvasEntityList.displayName = 'CanvasEntityList';
