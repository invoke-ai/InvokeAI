import { Flex } from '@invoke-ai/ui-library';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

const WorkflowLibraryListWrapper = (props: PropsWithChildren) => {
  return (
    <Flex w="full" h="full" flexDir="column" layerStyle="second" py={2} px={4} gap={2} borderRadius="base">
      {props.children}
    </Flex>
  );
};

export default memo(WorkflowLibraryListWrapper);
