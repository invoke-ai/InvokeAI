import { Heading } from '@invoke-ai/ui-library';
import { ElementWrapper } from 'features/nodes/components/sidePanel/builder/ElementWrapper';
import { useElement } from 'features/nodes/types/workflow';
import { memo } from 'react';

const LEVEL_TO_SIZE = {
  1: 'xl',
  2: 'lg',
  3: 'md',
  4: 'sm',
  5: 'xs',
} as const;

export const HeadingElementComponent = memo(({ id }: { id: string }) => {
  const element = useElement(id);

  if (!element || element.type !== 'heading') {
    return null;
  }
  const { data } = element;
  const { content, level } = data;

  return (
    <ElementWrapper id={id}>
      <Heading size={LEVEL_TO_SIZE[level]}>{content}</Heading>
    </ElementWrapper>
  );
});

HeadingElementComponent.displayName = 'HeadingElementComponent';
