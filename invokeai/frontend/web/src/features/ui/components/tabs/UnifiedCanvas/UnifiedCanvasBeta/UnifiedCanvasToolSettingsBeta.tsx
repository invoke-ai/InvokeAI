import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';

import { Flex } from '@chakra-ui/react';
import { isEqual } from 'lodash-es';
import UnifiedCanvasBaseBrushSettings from './UnifiedCanvasToolSettings/UnifiedCanvasBaseBrushSettings';
import UnifiedCanvasMaskBrushSettings from './UnifiedCanvasToolSettings/UnifiedCanvasMaskBrushSettings';
import UnifiedCanvasMoveSettings from './UnifiedCanvasToolSettings/UnifiedCanvasMoveSettings';

const selector = createSelector(
  [canvasSelector],
  (canvas) => {
    const { tool, layer } = canvas;
    return {
      tool,
      layer,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

export default function UnifiedCanvasToolSettingsBeta() {
  const { tool, layer } = useAppSelector(selector);

  return (
    <Flex height={8} minHeight={8} maxHeight={8} alignItems="center">
      {layer == 'base' && ['brush', 'eraser', 'colorPicker'].includes(tool) && (
        <UnifiedCanvasBaseBrushSettings />
      )}
      {layer == 'mask' && ['brush', 'eraser', 'colorPicker'].includes(tool) && (
        <UnifiedCanvasMaskBrushSettings />
      )}
      {tool == 'move' && <UnifiedCanvasMoveSettings />}
    </Flex>
  );
}
