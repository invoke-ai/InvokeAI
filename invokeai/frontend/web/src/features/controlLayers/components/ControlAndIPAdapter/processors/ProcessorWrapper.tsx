import { Flex } from '@invoke-ai/ui-library';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

type Props = PropsWithChildren;

const ProcessorWrapper = (props: Props) => {
  return (
    <Flex flexDir="column" gap={3}>
      {props.children}
    </Flex>
  );
};

export default memo(ProcessorWrapper);
