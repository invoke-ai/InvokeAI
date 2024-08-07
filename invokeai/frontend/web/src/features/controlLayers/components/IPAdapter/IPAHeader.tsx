import { Spacer } from '@invoke-ai/ui-library';
import { CanvasEntityDeleteButton } from 'features/controlLayers/components/common/CanvasEntityDeleteButton';
import { CanvasEntityEnabledToggle } from 'features/controlLayers/components/common/CanvasEntityEnabledToggle';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntityTitle } from 'features/controlLayers/components/common/CanvasEntityTitle';
import { memo } from 'react';

type Props = {
  onToggleVisibility: () => void;
};

export const IPAHeader = memo(({ onToggleVisibility }: Props) => {
  return (
    <CanvasEntityHeader onToggle={onToggleVisibility}>
      <CanvasEntityEnabledToggle />
      <CanvasEntityTitle />
      <Spacer />
      <CanvasEntityDeleteButton />
    </CanvasEntityHeader>
  );
});

IPAHeader.displayName = 'IPAHeader';
