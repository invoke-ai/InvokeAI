import type { FlexProps } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import { useContainerDirectionContext } from 'features/nodes/components/sidePanel/builder/ContainerContext';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

export const ElementWrapper = memo((props: PropsWithChildren<FlexProps>) => {
  const containerDirection = useContainerDirectionContext();
  return <Flex flex={containerDirection === 'column' ? '1 1 0' : undefined} {...props} />;
});

ElementWrapper.displayName = 'ElementWrapper';
