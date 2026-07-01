import { Flex } from '@invoke-ai/ui-library';
import { CanvasEntityDeleteButton } from 'features/controlLayers/components/common/CanvasEntityDeleteButton';
import { CanvasEntityEnabledToggle } from 'features/controlLayers/components/common/CanvasEntityEnabledToggle';
import { CanvasEntityHeaderWarnings } from 'features/controlLayers/components/common/CanvasEntityHeaderWarnings';
import { CanvasEntityIsBookmarkedForQuickSwitchToggle } from 'features/controlLayers/components/common/CanvasEntityIsBookmarkedForQuickSwitchToggle';
import { CanvasEntityIsLockedToggle } from 'features/controlLayers/components/common/CanvasEntityIsLockedToggle';
import { memo } from 'react';

export const CanvasEntityHeaderCommonActions = memo(() => {
  return (
    <Flex alignSelf="stretch">
      <CanvasEntityHeaderWarnings />
      <CanvasEntityIsBookmarkedForQuickSwitchToggle />
      <CanvasEntityIsLockedToggle />
      <CanvasEntityEnabledToggle />
      <CanvasEntityDeleteButton />
    </Flex>
  );
});

CanvasEntityHeaderCommonActions.displayName = 'CanvasEntityHeaderCommonActions';
