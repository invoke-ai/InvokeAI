import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import { DIVIDER_CLASS_NAME } from 'features/nodes/types/workflow';
import { memo } from 'react';

const sx: SystemStyleObject = {
  bg: 'base.700',
  flexShrink: 0,
};

export const DividerElementComponent = memo(({ id }: { id: string }) => {
  return <Flex id={id} className={DIVIDER_CLASS_NAME} sx={sx} />;
});

DividerElementComponent.displayName = 'DividerElementComponent';
