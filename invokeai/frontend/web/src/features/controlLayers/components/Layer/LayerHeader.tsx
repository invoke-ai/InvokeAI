import { Spacer } from '@invoke-ai/ui-library';
import { CanvasEntityDeleteButton } from 'features/controlLayers/components/common/CanvasEntityDeleteButton';
import { CanvasEntityEnabledToggle } from 'features/controlLayers/components/common/CanvasEntityEnabledToggle';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntityTitle } from 'features/controlLayers/components/common/CanvasEntityTitle';
import { LayerActionsMenu } from 'features/controlLayers/components/Layer/LayerActionsMenu';
import { memo } from 'react';

import { LayerOpacity } from './LayerOpacity';

type Props = {
  onToggleVisibility: () => void;
};

export const LayerHeader = memo(({ onToggleVisibility }: Props) => {
  return (
    <CanvasEntityHeader onToggle={onToggleVisibility}>
      <CanvasEntityEnabledToggle />
      <CanvasEntityTitle />
      <Spacer />
      <LayerOpacity />
      <LayerActionsMenu />
      <CanvasEntityDeleteButton />
    </CanvasEntityHeader>
  );
});

LayerHeader.displayName = 'LayerHeader';
