import { CanvasEntitySettings } from 'features/controlLayers/components/common/CanvasEntitySettings';
import { LayerControlAdapter } from 'features/controlLayers/components/Layer/LayerControlAdapter';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useLayerControlAdapter } from 'features/controlLayers/hooks/useLayerControlAdapter';
import { memo } from 'react';

export const LayerSettings = memo(() => {
  const entityIdentifier = useEntityIdentifierContext();
  const controlAdapter = useLayerControlAdapter(entityIdentifier);

  if (!controlAdapter) {
    return null;
  }

  return (
    <CanvasEntitySettings>
      <LayerControlAdapter controlAdapter={controlAdapter} />
    </CanvasEntitySettings>
  );
});

LayerSettings.displayName = 'LayerSettings';
