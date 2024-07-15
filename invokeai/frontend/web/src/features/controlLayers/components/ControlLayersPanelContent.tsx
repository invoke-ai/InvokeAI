/* eslint-disable i18next/no-literal-string */
import { Flex } from '@invoke-ai/ui-library';
import { AddLayerButton } from 'features/controlLayers/components/AddLayerButton';
import { CanvasEntityList } from 'features/controlLayers/components/CanvasEntityList';
import { DeleteAllLayersButton } from 'features/controlLayers/components/DeleteAllLayersButton';
import { memo } from 'react';

export const ControlLayersPanelContent = memo(() => {
  return (
    <Flex flexDir="column" gap={2} w="full" h="full">
      <Flex justifyContent="space-around">
        <AddLayerButton />
        <DeleteAllLayersButton />
      </Flex>
      <CanvasEntityList />
    </Flex>
  );
});

ControlLayersPanelContent.displayName = 'ControlLayersPanelContent';
