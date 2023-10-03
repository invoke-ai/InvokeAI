import { Flex } from '@chakra-ui/layout';
import { useAppSelector } from 'app/store/storeHooks';
import { memo } from 'react';
import FieldTypeLegend from './FieldTypeLegend';
import WorkflowEditorSettings from './WorkflowEditorSettings';

const TopRightPanel = () => {
  const shouldShowFieldTypeLegend = useAppSelector(
    (state) => state.nodes.shouldShowFieldTypeLegend
  );

  return (
    <Flex sx={{ gap: 2, position: 'absolute', top: 2, insetInlineEnd: 2 }}>
      <WorkflowEditorSettings />
      {shouldShowFieldTypeLegend && <FieldTypeLegend />}
    </Flex>
  );
};

export default memo(TopRightPanel);
