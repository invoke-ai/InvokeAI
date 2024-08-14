/* eslint-disable i18next/no-literal-string */
import { Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { AddLayerButton } from 'features/controlLayers/components/AddLayerButton';
import { CanvasEntityList } from 'features/controlLayers/components/CanvasEntityList';
import { DeleteAllLayersButton } from 'features/controlLayers/components/DeleteAllLayersButton';
import { Filter } from 'features/controlLayers/components/Filters/Filter';
import { $filteringEntity } from 'features/controlLayers/store/canvasV2Slice';
import ResizeHandle from 'features/ui/components/tabs/ResizeHandle';
import { memo } from 'react';
import { Panel, PanelGroup } from 'react-resizable-panels';

export const ControlLayersPanelContent = memo(() => {
  const filteringEntity = useStore($filteringEntity);
  return (
    <PanelGroup direction="vertical">
      <Panel id="canvas-entity-list-panel" order={0}>
        <Flex flexDir="column" gap={2} w="full" h="full">
          <Flex justifyContent="space-around">
            <AddLayerButton />
            <DeleteAllLayersButton />
          </Flex>
          <CanvasEntityList />
        </Flex>
      </Panel>
      {Boolean(filteringEntity) && (
        <>
          <ResizeHandle orientation="horizontal" />
          <Panel id="filter-panel" order={1}>
            <Filter />
          </Panel>
        </>
      )}
    </PanelGroup>
  );
});

ControlLayersPanelContent.displayName = 'ControlLayersPanelContent';
