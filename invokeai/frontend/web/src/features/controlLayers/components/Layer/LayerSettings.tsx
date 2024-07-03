import { CanvasEntitySettings } from 'features/controlLayers/components/common/CanvasEntitySettings';
import { memo } from 'react';

type Props = {
  id: string;
};

export const LayerSettings = memo(({ id }: Props) => {
  return <CanvasEntitySettings>PLACEHOLDER</CanvasEntitySettings>;
});

LayerSettings.displayName = 'LayerSettings';
