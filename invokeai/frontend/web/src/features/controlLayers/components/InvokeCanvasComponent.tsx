import { Box } from '@invoke-ai/ui-library';
import { useInvokeCanvas } from 'features/controlLayers/hooks/useInvokeCanvas';
import { memo } from 'react';

export const InvokeCanvasComponent = memo(() => {
  const ref = useInvokeCanvas();

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
