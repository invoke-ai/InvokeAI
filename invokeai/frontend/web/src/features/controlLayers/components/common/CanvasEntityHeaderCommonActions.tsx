import { Flex } from '@invoke-ai/ui-library';
import { CanvasEntityDeleteButton } from 'features/controlLayers/components/common/CanvasEntityDeleteButton';
import { CanvasEntityEnabledToggle } from 'features/controlLayers/components/common/CanvasEntityEnabledToggle';
import { CanvasEntityIsBookmarkedForQuickSwitchToggle } from 'features/controlLayers/components/common/CanvasEntityIsBookmarkedForQuickSwitchToggle';
import { CanvasEntityIsLockedToggle } from 'features/controlLayers/components/common/CanvasEntityIsLockedToggle';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { memo } from 'react';

import { CanvasEntityCopyToClipboard } from './CanvasEntityCopyToClipboard';

export const CanvasEntityHeaderCommonActions = memo(() => {
  const entityIdentifier = useEntityIdentifierContext();

  return (
    <Flex alignSelf="stretch">
      <CanvasEntityCopyToClipboard />
      <CanvasEntityIsBookmarkedForQuickSwitchToggle />
      {entityIdentifier.type !== 'reference_image' && <CanvasEntityIsLockedToggle />}
      <CanvasEntityEnabledToggle />
      <CanvasEntityDeleteButton />
    </Flex>
  );
});

CanvasEntityHeaderCommonActions.displayName = 'CanvasEntityHeaderCommonActions';
