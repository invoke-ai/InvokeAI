import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import { useContainerContext } from 'features/nodes/components/sidePanel/builder/contexts';
import { TextElementContent } from 'features/nodes/components/sidePanel/builder/TextElementContent';
import { TEXT_CLASS_NAME, type TextElement } from 'features/nodes/types/workflow';
import { memo } from 'react';

const sx: SystemStyleObject = {
  '&[data-parent-layout="column"]': {
    w: 'full',
    h: 'min-content',
  },
  '&[data-parent-layout="row"]': {
    flex: '1 1 0',
    minW: 32,
  },
};

export const TextElementViewMode = memo(({ el }: { el: TextElement }) => {
  const { id, data } = el;
  const { content } = data;
  const containerCtx = useContainerContext();

  return (
    <Flex id={id} className={TEXT_CLASS_NAME} sx={sx} data-parent-layout={containerCtx.layout}>
      <TextElementContent content={content} />
    </Flex>
  );
});

TextElementViewMode.displayName = 'TextElementViewMode';
