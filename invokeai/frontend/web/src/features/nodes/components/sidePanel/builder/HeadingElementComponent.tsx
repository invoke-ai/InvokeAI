import { Flex, Heading } from '@invoke-ai/ui-library';
import { useElement } from 'features/nodes/store/workflowSlice';
import { HEADING_CLASS_NAME, isHeadingElement } from 'features/nodes/types/workflow';
import { memo } from 'react';

const LEVEL_TO_SIZE = {
  1: 'xl',
  2: 'lg',
  3: 'md',
  4: 'sm',
  5: 'xs',
} as const;

export const HeadingElementComponent = memo(({ id }: { id: string }) => {
  const el = useElement(id);

  if (!el || !isHeadingElement(el)) {
    return null;
  }

  const { content, level } = el.data;

  return (
    <Flex id={id} className={HEADING_CLASS_NAME}>
      <Heading size={LEVEL_TO_SIZE[level]}>{content}</Heading>
    </Flex>
  );
});

HeadingElementComponent.displayName = 'HeadingElementComponent';
