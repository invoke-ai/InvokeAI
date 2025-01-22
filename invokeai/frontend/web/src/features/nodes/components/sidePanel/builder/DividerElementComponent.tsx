import { Flex } from '@invoke-ai/ui-library';
import type { DividerElement } from 'features/nodes/types/workflow';
import { memo } from 'react';

export const DividerElementComponent = memo(({ element }: { element: DividerElement }) => {
  const { id } = element;

  return <Flex id={id} h="1px" bg="base.800" flexShrink={0} />;
});

DividerElementComponent.displayName = 'DividerElementComponent';
