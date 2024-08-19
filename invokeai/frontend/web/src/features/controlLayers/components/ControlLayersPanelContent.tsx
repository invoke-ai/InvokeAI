/* eslint-disable i18next/no-literal-string */
import { useStore } from '@nanostores/react';
import { CanvasEntityList } from 'features/controlLayers/components/CanvasEntityList';
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
        <CanvasEntityList />
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
