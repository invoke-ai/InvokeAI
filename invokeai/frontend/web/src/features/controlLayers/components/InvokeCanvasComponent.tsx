import { Box } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useInvokeCanvas } from 'features/controlLayers/hooks/useInvokeCanvas';
import { selectActiveCanvasId } from 'features/controlLayers/store/selectors';
import { memo } from 'react';

export const InvokeCanvasComponent = memo(() => {
  const canvasId = useAppSelector(selectActiveCanvasId);
  const ref = useInvokeCanvas(canvasId);

  return (
    <Box
      position="absolute"
      top={0}
      right={0}
      bottom={0}
      left={0}
      ref={ref}
      borderRadius="base"
      overflow="hidden"
      data-testid="control-layers-canvas"
    />
  );
});

InvokeCanvasComponent.displayName = 'InvokeCanvasComponent';
