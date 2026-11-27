import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import { useContainerContext } from 'features/nodes/components/sidePanel/builder/contexts';
import { HeadingElementContent } from 'features/nodes/components/sidePanel/builder/HeadingElementContent';
import type { HeadingElement } from 'features/nodes/types/workflow';
import { HEADING_CLASS_NAME } from 'features/nodes/types/workflow';
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

export const HeadingElementViewMode = memo(({ el }: { el: HeadingElement }) => {
  const { id, data } = el;
  const { content } = data;
  const containerCtx = useContainerContext();

  return (
    <Flex id={id} className={HEADING_CLASS_NAME} sx={sx} data-parent-layout={containerCtx.layout}>
      <HeadingElementContent content={content} />
    </Flex>
  );
});

HeadingElementViewMode.displayName = 'HeadingElementViewMode';
