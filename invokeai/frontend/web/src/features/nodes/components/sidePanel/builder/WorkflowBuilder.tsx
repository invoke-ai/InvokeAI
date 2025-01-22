import { Flex } from '@invoke-ai/ui-library';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { FormElementComponent } from 'features/nodes/components/sidePanel/builder/ContainerElementComponent';
import { rootId } from 'features/nodes/types/workflow';
import { memo } from 'react';

export const WorkflowBuilder = memo(() => {
  return (
    <ScrollableContent>
      <Flex w="full" justifyContent="center">
        <Flex w="full" maxW={512}>
          <FormElementComponent id={rootId} />
        </Flex>
      </Flex>
    </ScrollableContent>
  );
});

WorkflowBuilder.displayName = 'WorkflowBuilder';
