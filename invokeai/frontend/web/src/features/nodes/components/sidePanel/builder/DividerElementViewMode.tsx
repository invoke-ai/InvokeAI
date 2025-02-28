import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import { useContainerContext } from 'features/nodes/components/sidePanel/builder/contexts';
import type { DividerElement } from 'features/nodes/types/workflow';
import { DIVIDER_CLASS_NAME } from 'features/nodes/types/workflow';
import { memo } from 'react';

const sx: SystemStyleObject = {
  bg: 'base.700',
  flexShrink: 0,
  '&[data-layout="column"]': {
    width: '100%',
    height: '1px',
  },
  '&[data-layout="row"]': {
    height: '100%',
    width: '1px',
  },
};

export const DividerElementViewMode = memo(({ el }: { el: DividerElement }) => {
  const container = useContainerContext();
  const { id } = el;

  return (
    <Flex
      id={id}
      className={DIVIDER_CLASS_NAME}
      sx={sx}
      data-layout={
        // When there is no container, the layout is column by default
        container?.layout || 'column'
      }
    />
  );
});

DividerElementViewMode.displayName = 'DividerElementViewMode';
