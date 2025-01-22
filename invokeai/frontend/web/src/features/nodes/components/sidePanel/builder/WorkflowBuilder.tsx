import { Flex } from '@invoke-ai/ui-library';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { ContainerElementComponent } from 'features/nodes/components/sidePanel/builder/ContainerElementComponent';
import { data } from 'features/nodes/types/workflow';
import { memo } from 'react';

export const WorkflowBuilder = memo(() => {
  return (
    <ScrollableContent>
      <Flex w="full" h="full" justifyContent="center">
        <Flex w="full" h="full" maxW={512}>
          <ContainerElementComponent element={data} />
        </Flex>
      </Flex>
    </ScrollableContent>
  );
});

WorkflowBuilder.displayName = 'WorkflowBuilder';
