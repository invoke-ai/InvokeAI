import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import { HeadingElementContent } from 'features/nodes/components/sidePanel/builder/HeadingElementContent';
import type { HeadingElement } from 'features/nodes/types/workflow';
import { HEADING_CLASS_NAME } from 'features/nodes/types/workflow';
import { memo } from 'react';

const sx: SystemStyleObject = {
  flex: '0 1 0',
};

export const HeadingElementViewMode = memo(({ el }: { el: HeadingElement }) => {
  const { id, data } = el;
  const { content } = data;

  return (
    <Flex id={id} className={HEADING_CLASS_NAME} sx={sx}>
      <HeadingElementContent content={content} />
    </Flex>
  );
});

HeadingElementViewMode.displayName = 'HeadingElementViewMode';
