import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import { useContainerContext } from 'features/nodes/components/sidePanel/builder/contexts';
import { memo } from 'react';

const sx: SystemStyleObject = {
  bg: 'base.700',
  flexShrink: 0,
  '&[data-parent-layout="column"]': {
    width: '100%',
    height: '1px',
  },
  '&[data-parent-layout="row"]': {
    height: '100%',
    width: '1px',
  },
};

export const DividerElementComponent = memo(() => {
  const containerCtx = useContainerContext();

  return <Flex sx={sx} data-parent-layout={containerCtx.layout} />;
});

DividerElementComponent.displayName = 'DividerElementComponent';
