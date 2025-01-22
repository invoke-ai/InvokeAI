import { Flex, Text } from '@invoke-ai/ui-library';
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
    <Flex id={id}>
      <Text fontSize={fontSize}>{content}</Text>
    </Flex>
  );
});

TextElementComponent.displayName = 'TextElementComponent';
