import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { memo } from 'react';

export const DIVIDER_CLASS_NAME = getPrefixedId('divider');

const sx: SystemStyleObject = {
  flex: '0 0 1px',
  bg: 'base.700',
  flexShrink: 0,
};

export const DividerElementComponent = memo(({ id }: { id: string }) => {
  return <Flex id={id} className={DIVIDER_CLASS_NAME} sx={sx} />;
});

DividerElementComponent.displayName = 'DividerElementComponent';
