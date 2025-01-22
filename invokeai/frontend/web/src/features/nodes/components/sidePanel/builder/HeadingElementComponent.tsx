import { Heading } from '@invoke-ai/ui-library';
import type { HeadingElement } from 'features/nodes/types/workflow';
import { memo } from 'react';

const LEVEL_TO_SIZE = {
  1: 'xl',
  2: 'lg',
  3: 'md',
  4: 'sm',
  5: 'xs',
} as const;

export const HeadingElementComponent = memo(({ element }: { element: HeadingElement }) => {
  const { id, data } = element;
  const { content, level } = data;

  return (
    <Heading id={id} size={LEVEL_TO_SIZE[level]}>
      {content}
    </Heading>
  );
});

HeadingElementComponent.displayName = 'HeadingElementComponent';
