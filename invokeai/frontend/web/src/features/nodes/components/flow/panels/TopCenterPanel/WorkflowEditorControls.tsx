import { Flex } from '@chakra-ui/react';
import CancelButton from 'features/parameters/components/ProcessButtons/CancelButton';
import InvokeButton from 'features/parameters/components/ProcessButtons/InvokeButton';
import { memo } from 'react';

const WorkflowEditorControls = () => {
  return (
    <Flex layerStyle="first" sx={{ gap: 2, borderRadius: 'base', p: 2 }}>
      <InvokeButton />
      <CancelButton />
    </Flex>
  );
};

export default memo(WorkflowEditorControls);
