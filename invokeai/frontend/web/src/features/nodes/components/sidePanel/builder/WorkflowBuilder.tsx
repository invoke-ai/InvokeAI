import { Flex } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { FormElementComponent } from 'features/nodes/components/sidePanel/builder/ContainerElementComponent';
import { formLoaded } from 'features/nodes/store/workflowSlice';
import { elements, rootElementId } from 'features/nodes/types/workflow';
import { memo, useEffect } from 'react';

export const WorkflowBuilder = memo(() => {
  const dispatch = useAppDispatch();
  useEffect(() => {
    dispatch(formLoaded({ elements, rootElementId }));
  }, [dispatch]);
  return (
    <ScrollableContent>
      <Flex w="full" justifyContent="center">
        <Flex w="full" maxW={512}>
          {rootElementId && <FormElementComponent id={rootElementId} />}
        </Flex>
      </Flex>
    </ScrollableContent>
  );
});

WorkflowBuilder.displayName = 'WorkflowBuilder';
