import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { LineComponent } from 'features/regionalPrompts/components/LineComponent';
import { RectComponent } from 'features/regionalPrompts/components/RectComponent';
import {
  layerObjectsSelectors,
  layersSelectors,
  selectRegionalPromptsSlice,
} from 'features/regionalPrompts/store/regionalPromptsSlice';
import { memo } from 'react';
import { Group, Layer, Stage } from 'react-konva';

const selectLayers = createSelector(selectRegionalPromptsSlice, (regionalPrompts) =>
  layersSelectors.selectAll(regionalPrompts)
);

export const RegionalPromptsStage: React.FC = memo(() => {
  const layers = useAppSelector(selectLayers);
  return (
    <Stage width={window.innerWidth} height={window.innerHeight}>
      <Layer>
        {layers.map((layer) => (
          <Group key={layer.id}>
            {layerObjectsSelectors.selectAll(layer.objects).map((obj) => {
              if (obj.kind === 'line') {
                return <LineComponent key={obj.id} line={obj} />;
              }
              if (obj.kind === 'fillRect') {
                return <RectComponent key={obj.id} rect={obj} />;
              }
            })}
          </Group>
        ))}
      </Layer>
    </Stage>
  );
});

RegionalPromptsStage.displayName = 'RegionalPromptingEditor';
