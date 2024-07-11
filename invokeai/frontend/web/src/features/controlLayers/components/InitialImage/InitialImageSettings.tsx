import { CanvasEntitySettings } from 'features/controlLayers/components/common/CanvasEntitySettings';
import { InitialImagePreview } from 'features/controlLayers/components/InitialImage/InitialImagePreview';
import { memo } from 'react';

export const InitialImageSettings = memo(() => {
  return (
    <CanvasEntitySettings>
      <InitialImagePreview />
    </CanvasEntitySettings>
  );
});

InitialImageSettings.displayName = 'InitialImageSettings';
