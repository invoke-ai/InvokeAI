import { chakra } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { BrushPreviewOutline } from 'features/regionalPrompts/components/BrushPreview';
import { LayerComponent } from 'features/regionalPrompts/components/LayerComponent';
import {
  useMouseDown,
  useMouseEnter,
  useMouseLeave,
  useMouseMove,
  useMouseUp,
} from 'features/regionalPrompts/hooks/mouseEventHooks';
import { $stage, selectRegionalPromptsSlice } from 'features/regionalPrompts/store/regionalPromptsSlice';
import type Konva from 'konva';
import { memo, useCallback, useMemo, useRef } from 'react';
import { Layer, Stage } from 'react-konva';

const selectLayerIds = createSelector(selectRegionalPromptsSlice, (regionalPrompts) =>
  regionalPrompts.layers.map((l) => l.id)
);

const ChakraStage = chakra(Stage, {
  shouldForwardProp: (prop) => !['sx'].includes(prop),
});

export const RegionalPromptsStage: React.FC = memo(() => {
  const layerIds = useAppSelector(selectLayerIds);
  const stageRef = useRef<Konva.Stage | null>(null);
  const width = useAppSelector((s) => s.generation.width);
  const height = useAppSelector((s) => s.generation.height);
  const tool = useAppSelector((s) => s.regionalPrompts.tool);
  const onMouseDown = useMouseDown(stageRef);
  const onMouseUp = useMouseUp(stageRef);
  const onMouseMove = useMouseMove(stageRef);
  const onMouseEnter = useMouseEnter(stageRef);
  const onMouseLeave = useMouseLeave(stageRef);
  const stageRefCallback = useCallback((el: Konva.Stage) => {
    $stage.set(el);
    stageRef.current = el;
  }, []);
  const sx = useMemo(
    () => ({
      border: '1px solid cyan',
      cursor: tool === 'move' ? 'default' : 'none',
    }),
    [tool]
  );

  return (
    <ChakraStage
      ref={stageRefCallback}
      x={0}
      y={0}
      width={width}
      height={height}
      onMouseDown={onMouseDown}
      onMouseUp={onMouseUp}
      onMouseMove={onMouseMove}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      tabIndex={-1}
      sx={sx}
    >
      {layerIds.map((id) => (
        <LayerComponent key={id} id={id} />
      ))}
      <Layer id="brushPreviewOutline">
        <BrushPreviewOutline />
      </Layer>
    </ChakraStage>
  );
});

RegionalPromptsStage.displayName = 'RegionalPromptsStage';
