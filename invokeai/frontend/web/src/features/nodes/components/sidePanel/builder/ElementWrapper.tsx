import type { FlexProps, SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex } from '@invoke-ai/ui-library';
import { useContainerContext } from 'features/nodes/components/sidePanel/builder/ContainerContext';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

const sx: SystemStyleObject = {
  // '&[data-container-direction="column"]': {
  //   flex: '1 1 auto',
  // },
  // '&[data-container-direction="column"] > :not(:last-child)': {
  //   bg: 'red',
  // },
};

export const ElementWrapper = memo((props: PropsWithChildren<FlexProps>) => {
  const container = useContainerContext();
  return (
    <Flex
      sx={sx}
      // data-container-direction={container?.direction}
      {...props}
    />
  );
});

ElementWrapper.displayName = 'ElementWrapper';
