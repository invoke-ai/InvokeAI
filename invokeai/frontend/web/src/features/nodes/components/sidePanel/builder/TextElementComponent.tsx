import { Text } from '@invoke-ai/ui-library';
import type { TextElement } from 'features/nodes/types/workflow';
import { memo } from 'react';

export const TextElementComponent = memo(({ element }: { element: TextElement }) => {
  const { id, data } = element;
  const { content, fontSize } = data;

  return (
    <Text id={id} fontSize={fontSize}>
      {content}
    </Text>
  );
});

TextElementComponent.displayName = 'TextElementComponent';
