import { Flex } from '@invoke-ai/ui-library';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

const StyledFlex = (props: PropsWithChildren) => {
  return (
    <Flex flexDirection="column" gap={2} p={4} borderRadius="base" bg="base.900">
      {props.children}
    </Flex>
  );
};

export default memo(StyledFlex);
