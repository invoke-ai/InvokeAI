import { Spacer } from '@invoke-ai/ui-library';
import { CanvasEntityEnabledToggle } from 'features/controlLayers/components/common/CanvasEntityEnabledToggle';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntityTitle } from 'features/controlLayers/components/common/CanvasEntityTitle';
import { IMActionsMenu } from 'features/controlLayers/components/InpaintMask/IMActionsMenu';
import { memo } from 'react';

import { IMMaskFillColorPicker } from './IMMaskFillColorPicker';

type Props = {
  onToggleVisibility: () => void;
};

export const IMHeader = memo(({ onToggleVisibility }: Props) => {
  return (
    <CanvasEntityHeader onToggle={onToggleVisibility}>
      <CanvasEntityEnabledToggle />
      <CanvasEntityTitle />
      <Spacer />
      <IMMaskFillColorPicker />
      <IMActionsMenu />
    </CanvasEntityHeader>
  );
});

IMHeader.displayName = 'IMHeader';
