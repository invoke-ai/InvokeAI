import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { layerSelected, selectRegionalPromptsSlice } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { useCallback, useMemo } from 'react';
import { Rect as KonvaRect } from 'react-konva';

type Props = {
  layerId: string;
};

export const LayerBoundingBox = ({ layerId }: Props) => {
  const dispatch = useAppDispatch();
  const tool = useAppSelector((s) => s.regionalPrompts.tool);
  const selectedLayer = useAppSelector((s) => s.regionalPrompts.selectedLayer);

  const onMouseDown = useCallback(() => {
    dispatch(layerSelected(layerId));
  }, [dispatch, layerId]);

  const selectBbox = useMemo(
    () =>
      createSelector(
        selectRegionalPromptsSlice,
        (regionalPrompts) => regionalPrompts.layers.find((layer) => layer.id === layerId)?.bbox ?? null
      ),
    [layerId]
  );
  const bbox = useAppSelector(selectBbox);

  if (!bbox || tool !== 'move') {
    return null;
  }

  return (
    <KonvaRect
      onMouseDown={onMouseDown}
      stroke={selectedLayer === layerId ? 'rgba(0, 238, 255, 1)' : 'rgba(255,255,255,0.3)'}
      strokeWidth={1}
      x={bbox.x}
      y={bbox.y}
      width={bbox.width}
      height={bbox.height}
      listening={tool === 'move'}
    />
  );
};
