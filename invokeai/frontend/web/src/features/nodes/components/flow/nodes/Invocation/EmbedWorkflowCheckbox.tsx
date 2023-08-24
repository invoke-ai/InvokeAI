import { Checkbox, Flex, FormControl, FormLabel } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { useEmbedWorkflow } from 'features/nodes/hooks/useEmbedWorkflow';
import { useHasImageOutput } from 'features/nodes/hooks/useHasImageOutput';
import { nodeEmbedWorkflowChanged } from 'features/nodes/store/nodesSlice';
import { ChangeEvent, memo, useCallback } from 'react';

const EmbedWorkflowCheckbox = ({ nodeId }: { nodeId: string }) => {
  const dispatch = useAppDispatch();
  const hasImageOutput = useHasImageOutput(nodeId);
  const embedWorkflow = useEmbedWorkflow(nodeId);
  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(
        nodeEmbedWorkflowChanged({
          nodeId,
          embedWorkflow: e.target.checked,
        })
      );
    },
    [dispatch, nodeId]
  );

  if (!hasImageOutput) {
    return null;
  }

  return (
    <FormControl as={Flex} sx={{ alignItems: 'center', gap: 2, w: 'auto' }}>
      <FormLabel sx={{ fontSize: 'xs', mb: '1px' }}>Embed Workflow</FormLabel>
      <Checkbox
        className="nopan"
        size="sm"
        onChange={handleChange}
        isChecked={embedWorkflow}
      />
    </FormControl>
  );
};

export default memo(EmbedWorkflowCheckbox);
