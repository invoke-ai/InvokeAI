import { Flex } from '@invoke-ai/ui-library';
import { TextElementContent } from 'features/nodes/components/sidePanel/builder/TextElementContent';
import { TEXT_CLASS_NAME, type TextElement } from 'features/nodes/types/workflow';
import { memo } from 'react';

export const TextElementViewMode = memo(({ el }: { el: TextElement }) => {
  const { id, data } = el;
  const { content } = data;

  return (
    <Flex id={id} className={TEXT_CLASS_NAME} w="full" minW={32}>
      <TextElementContent content={content} />
    </Flex>
  );
});

TextElementViewMode.displayName = 'TextElementViewMode';
