import type { SpinnerProps } from '@invoke-ai/ui-library';
import { Spinner } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useCanvasManager } from 'features/controlLayers/hooks/useCanvasManager';
import { useAllEntityAdapters } from 'features/controlLayers/contexts/EntityAdapterContext';
import { computed } from 'nanostores';
import { memo, useMemo } from 'react';

export const CanvasBusySpinner = memo((props: SpinnerProps) => {
  const canvasManager = useCanvasManager();
  const allEntityAdapters = useAllEntityAdapters();
  const $isPendingRectCalculation = useMemo(
    () =>
      computed(
        allEntityAdapters.map(({ transformer }) => transformer.$isPendingRectCalculation),
        (...values) => values.some((v) => v)
      ),
    [allEntityAdapters]
  );
  const isPendingRectCalculation = useStore($isPendingRectCalculation);
  const isRasterizing = useStore(canvasManager.stateApi.$isRasterizing);
  const isCompositing = useStore(canvasManager.compositor.$isBusy);

  if (isRasterizing || isCompositing || isPendingRectCalculation) {
    return <Spinner opacity={0.3} {...props} />;
  }
  return null;
});
CanvasBusySpinner.displayName = 'CanvasBusySpinner';
