import { Spacer } from '@invoke-ai/ui-library';
import { CanvasEntityDeleteButton } from 'features/controlLayers/components/common/CanvasEntityDeleteButton';
import { CanvasEntityEnabledToggle } from 'features/controlLayers/components/common/CanvasEntityEnabledToggle';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntityTitle } from 'features/controlLayers/components/common/CanvasEntityTitle';
import { CAActionsMenu } from 'features/controlLayers/components/ControlAdapter/CAActionsMenu';
import { CAOpacityAndFilter } from 'features/controlLayers/components/ControlAdapter/CAOpacityAndFilter';
import { memo } from 'react';

type Props = {
  onToggleVisibility: () => void;
};

export const CAHeader = memo(({ onToggleVisibility }: Props) => {
  return (
    <CanvasEntityHeader onToggle={onToggleVisibility}>
      <CanvasEntityEnabledToggle />
      <CanvasEntityTitle />
      <Spacer />
      <CAOpacityAndFilter />
      <CAActionsMenu />
      <CanvasEntityDeleteButton />
    </CanvasEntityHeader>
  );
});

CAHeader.displayName = 'CAEntityHeader';
