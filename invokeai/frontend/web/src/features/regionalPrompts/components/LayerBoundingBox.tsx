import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { layerSelected, selectRegionalPromptsSlice } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { memo, useCallback, useMemo } from 'react';
import { Rect as KonvaRect } from 'react-konva';

type Props = {
  layerId: string;
};

export const LayerBoundingBox = memo(({ layerId }: Props) => {
  const dispatch = useAppDispatch();
  const tool = useAppSelector((s) => s.regionalPrompts.tool);
  const selectedLayer = useAppSelector((s) => s.regionalPrompts.selectedLayer);

  const onMouseDown = useCallback(() => {
    dispatch(layerSelected(layerId));
  }, [dispatch, layerId]);

  const selectBbox = useMemo(
    () =>
      createMemoizedSelector(
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
      name="layer bbox"
      onMouseDown={onMouseDown}
      stroke={selectedLayer === layerId ? 'rgba(153, 187, 189, 1)' : 'rgba(255, 255, 255, 0.149)'}
      strokeWidth={1}
      x={bbox.x}
      y={bbox.y}
      width={bbox.width}
      height={bbox.height}
      listening={tool === 'move'}
    />
  );
});

LayerBoundingBox.displayName = 'LayerBoundingBox';
