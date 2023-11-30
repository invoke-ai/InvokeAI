import { Flex } from '@chakra-ui/layout';
import { memo } from 'react';
import WorkflowEditorSettings from './WorkflowEditorSettings';

const TopRightPanel = () => {
  return (
    <Flex sx={{ gap: 2, position: 'absolute', top: 2, insetInlineEnd: 2 }}>
      <WorkflowEditorSettings />
    </Flex>
  );
};

export default memo(TopRightPanel);
