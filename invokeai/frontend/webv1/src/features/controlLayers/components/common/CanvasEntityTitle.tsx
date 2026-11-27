import type { TextProps } from '@invoke-ai/ui-library';
import { Text } from '@invoke-ai/ui-library';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useEntityIsSelected } from 'features/controlLayers/hooks/useEntityIsSelected';
import { useEntityTitle } from 'features/controlLayers/hooks/useEntityTitle';
import { memo } from 'react';

export const CanvasEntityTitle = memo((props: TextProps) => {
  const entityIdentifier = useEntityIdentifierContext();
  const isSelected = useEntityIsSelected(entityIdentifier);
  const title = useEntityTitle(entityIdentifier);

  return (
    <Text size="sm" fontWeight="semibold" userSelect="none" color={isSelected ? 'base.100' : 'base.300'} {...props}>
      {title}
    </Text>
  );
});

CanvasEntityTitle.displayName = 'CanvasEntityTitle';
