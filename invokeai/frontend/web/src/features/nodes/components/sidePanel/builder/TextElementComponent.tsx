import { Flex, Text } from '@invoke-ai/ui-library';
import { useElement } from 'features/nodes/store/workflowSlice';
import { isTextElement, TEXT_CLASS_NAME } from 'features/nodes/types/workflow';
import { memo } from 'react';

export const TextElementComponent = memo(({ id }: { id: string }) => {
  const el = useElement(id);

  if (!el || !isTextElement(el)) {
    return null;
  }

  const { content, fontSize } = el.data;

  return (
    <Flex id={id} className={TEXT_CLASS_NAME}>
      <Text fontSize={fontSize}>{content}</Text>
    </Flex>
  );
});

TextElementComponent.displayName = 'TextElementComponent';
