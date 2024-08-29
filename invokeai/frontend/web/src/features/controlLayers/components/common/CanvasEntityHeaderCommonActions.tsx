import { Flex } from '@invoke-ai/ui-library';
import { CanvasEntityDeleteButton } from 'features/controlLayers/components/common/CanvasEntityDeleteButton';
import { CanvasEntityEnabledToggle } from 'features/controlLayers/components/common/CanvasEntityEnabledToggle';
import { CanvasEntityIsLockedToggle } from 'features/controlLayers/components/common/CanvasEntityIsLockedToggle';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { memo } from 'react';

export const CanvasEntityHeaderCommonActions = memo(() => {
  const entityIdentifier = useEntityIdentifierContext();

  return (
    <Flex alignSelf="stretch">
      {entityIdentifier.type !== 'ip_adapter' && <CanvasEntityIsLockedToggle />}
      <CanvasEntityEnabledToggle />
      <CanvasEntityDeleteButton />
    </Flex>
  );
});

CanvasEntityHeaderCommonActions.displayName = 'CanvasEntityHeaderCommonActions';
