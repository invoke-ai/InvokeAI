import type { FlexProps, SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import { useDepthContext } from 'features/nodes/components/sidePanel/builder/contexts';
import { memo } from 'react';

const contentWrapperSx: SystemStyleObject = {
  w: 'full',
  h: 'full',
  borderWidth: 1,
  borderRadius: 'base',
  borderTopRadius: 'unset',
  borderTop: 'unset',
  borderColor: 'baseAlpha.250',
  '&[data-depth="0"]': { borderColor: 'baseAlpha.100' },
  '&[data-depth="1"]': { borderColor: 'baseAlpha.150' },
  '&[data-depth="2"]': { borderColor: 'baseAlpha.200' },
  '&[data-is-dragging="true"]': {
    opacity: 0.3,
  },
};

export const FormElementEditModeContent = memo(({ children, ...rest }: FlexProps) => {
  const depth = useDepthContext();
  return (
    <Flex sx={contentWrapperSx} data-depth={depth} {...rest}>
      {children}
    </Flex>
  );
});
FormElementEditModeContent.displayName = 'FormElementEditModeContent';
