import { useAppSelector } from 'app/store/storeHooks';
import { CanvasAddEntityButtons } from 'features/controlLayers/components/CanvasAddEntityButtons';
import { CanvasEntityList } from 'features/controlLayers/components/CanvasEntityList';
import { CanvasManagerProviderGate } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { selectEntityCount } from 'features/controlLayers/store/selectors';
import { memo } from 'react';

export const CanvasPanelContent = memo(() => {
  const hasEntities = useAppSelector((s) => selectEntityCount(s) > 0);
  return (
    <CanvasManagerProviderGate>
      {!hasEntities && <CanvasAddEntityButtons />}
      {hasEntities && <CanvasEntityList />}
    </CanvasManagerProviderGate>
  );
});

CanvasPanelContent.displayName = 'CanvasPanelContent';
