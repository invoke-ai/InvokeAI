/* eslint-disable i18next/no-literal-string */
import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { AddLayerButton } from 'features/controlLayers/components/AddLayerButton';
import { CanvasEntityList } from 'features/controlLayers/components/CanvasEntityList';
import { DeleteAllLayersButton } from 'features/controlLayers/components/DeleteAllLayersButton';
import { InitialImage } from 'features/controlLayers/components/InitialImage/InitialImage';
import { IM } from 'features/controlLayers/components/InpaintMask/IM';
import { memo } from 'react';

export const ControlLayersPanelContent = memo(() => {
  const isCanvasSessionActive = useAppSelector((s) => s.canvasV2.session.isActive);
  return (
    <Flex flexDir="column" gap={2} w="full" h="full">
      <Flex justifyContent="space-around">
        <AddLayerButton />
        <DeleteAllLayersButton />
      </Flex>
      {isCanvasSessionActive && <IM />}
      <CanvasEntityList />
      {!isCanvasSessionActive && <InitialImage />}
    </Flex>
  );
});

ControlLayersPanelContent.displayName = 'ControlLayersPanelContent';
