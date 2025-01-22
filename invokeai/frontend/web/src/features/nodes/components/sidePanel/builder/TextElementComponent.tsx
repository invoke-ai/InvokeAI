import { Text } from '@invoke-ai/ui-library';
import { ElementWrapper } from 'features/nodes/components/sidePanel/builder/ElementWrapper';
import { useElement } from 'features/nodes/types/workflow';
import { memo } from 'react';

export const TextElementComponent = memo(({ id }: { id: string }) => {
  const element = useElement(id);

  if (!element || element.type !== 'text') {
    return null;
  }
  const { data } = element;
  const { content, fontSize } = data;

  return (
    <ElementWrapper id={id}>
      <Text fontSize={fontSize}>{content}</Text>
    </ElementWrapper>
  );
});

TextElementComponent.displayName = 'TextElementComponent';
