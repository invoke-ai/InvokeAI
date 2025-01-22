import { Flex } from '@invoke-ai/ui-library';
import { useContainerDirectionContext } from 'features/nodes/components/sidePanel/builder/ContainerContext';
import { memo } from 'react';

export const DividerElementComponent = memo(({ id }: { id: string }) => {
  const containerDirection = useContainerDirectionContext();

  return (
    <Flex
      flex="0 0 auto"
      id={id}
      h={containerDirection === 'column' ? '1px' : undefined}
      w={containerDirection === 'column' ? undefined : '1px'}
      bg="base.700"
      flexShrink={0}
    />
  );
});

DividerElementComponent.displayName = 'DividerElementComponent';
