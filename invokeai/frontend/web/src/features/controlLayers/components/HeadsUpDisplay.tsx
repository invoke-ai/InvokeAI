import { Box, Flex, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';
import { round } from 'lodash-es';
import { memo } from 'react';

const selectBbox = createSelector(selectCanvasSlice, (canvas) => canvas.bbox);

export const HeadsUpDisplay = memo(() => {
  const canvasManager = useCanvasManager();
  const stageAttrs = useStore(canvasManager.stateApi.$stageAttrs);
  const cursorPos = useStore(canvasManager.stateApi.$lastCursorPos);
  const isDrawing = useStore(canvasManager.stateApi.$isDrawing);
  const isMouseDown = useStore(canvasManager.stateApi.$isMouseDown);
  const lastMouseDownPos = useStore(canvasManager.stateApi.$lastMouseDownPos);
  const lastAddedPoint = useStore(canvasManager.stateApi.$lastAddedPoint);
  const bbox = useAppSelector(selectBbox);

  return (
    <Flex flexDir="column" bg="blackAlpha.400" borderBottomEndRadius="base" p={2} minW={64} gap={2}>
      <HUDItem label="Zoom" value={`${round(stageAttrs.scale * 100, 2)}%`} />
      <HUDItem label="Stage Pos" value={`${round(stageAttrs.x, 3)}, ${round(stageAttrs.y, 3)}`} />
      <HUDItem
        label="Stage Size"
        value={`${round(stageAttrs.width / stageAttrs.scale, 2)}×${round(stageAttrs.height / stageAttrs.scale, 2)} px`}
      />
      <HUDItem label="BBox Size" value={`${bbox.rect.width}×${bbox.rect.height} px`} />
      <HUDItem label="BBox Position" value={`${bbox.rect.x}, ${bbox.rect.y}`} />
      <HUDItem label="BBox Width % 8" value={round(bbox.rect.width % 8, 2)} />
      <HUDItem label="BBox Height % 8" value={round(bbox.rect.height % 8, 2)} />
      <HUDItem label="BBox X % 8" value={round(bbox.rect.x % 8, 2)} />
      <HUDItem label="BBox Y % 8" value={round(bbox.rect.y % 8, 2)} />
      <HUDItem
        label="Cursor Position"
        value={cursorPos ? `${round(cursorPos.x, 2)}, ${round(cursorPos.y, 2)}` : '?, ?'}
      />
      <HUDItem label="Is Drawing" value={isDrawing ? 'True' : 'False'} />
      <HUDItem label="Is Mouse Down" value={isMouseDown ? 'True' : 'False'} />
      <HUDItem
        label="Last Mouse Down Pos"
        value={lastMouseDownPos ? `${round(lastMouseDownPos.x, 2)}, ${round(lastMouseDownPos.y, 2)}` : '?, ?'}
      />
      <HUDItem
        label="Last Added Point"
        value={lastAddedPoint ? `${round(lastAddedPoint.x, 2)}, ${round(lastAddedPoint.y, 2)}` : '?, ?'}
      />
    </Flex>
  );
});

HeadsUpDisplay.displayName = 'HeadsUpDisplay';

const HUDItem = memo(({ label, value }: { label: string; value: string | number }) => {
  return (
    <Box display="inline-block" lineHeight={1}>
      <Text as="span">{label}: </Text>
      <Text as="span" fontWeight="semibold">
        {value}
      </Text>
    </Box>
  );
});

HUDItem.displayName = 'HUDItem';