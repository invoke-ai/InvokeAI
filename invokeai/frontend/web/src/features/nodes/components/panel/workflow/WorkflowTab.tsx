import { Flex } from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import ImageMetadataJSON from 'features/gallery/components/ImageMetadataViewer/ImageMetadataJSON';
import { buildWorkflow } from 'features/nodes/util/buildWorkflow';
import { memo, useMemo } from 'react';
import { useDebounce } from 'use-debounce';

const useWatchWorkflow = () => {
  const nodes = useAppSelector((state: RootState) => state.nodes);
  const [debouncedNodes] = useDebounce(nodes, 300);
  const workflow = useMemo(
    () => buildWorkflow(debouncedNodes),
    [debouncedNodes]
  );

  return {
    workflow,
  };
};

const WorkflowWorkflowTab = () => {
  const { workflow } = useWatchWorkflow();

  return (
    <Flex
      sx={{
        flexDir: 'column',
        alignItems: 'flex-start',
        gap: 2,
        h: 'full',
      }}
    >
      <ImageMetadataJSON
        jsonObject={workflow}
        label="Workflow"
        fileName={workflow.name}
      />
    </Flex>
  );
};

export default memo(WorkflowWorkflowTab);
