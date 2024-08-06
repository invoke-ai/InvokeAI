import { CanvasEntitySettings } from 'features/controlLayers/components/common/CanvasEntitySettings';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { memo } from 'react';

export const LayerSettings = memo(() => {
  const entityIdentifier = useEntityIdentifierContext();
  return <CanvasEntitySettings>PLACEHOLDER</CanvasEntitySettings>;
});

LayerSettings.displayName = 'LayerSettings';
