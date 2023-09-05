import { FormControl, FormLabel } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import IAITextarea from 'common/components/IAITextarea';
import { useNodeData } from 'features/nodes/hooks/useNodeData';
import { nodeNotesChanged } from 'features/nodes/store/nodesSlice';
import { isInvocationNodeData } from 'features/nodes/types/types';
import { ChangeEvent, memo, useCallback } from 'react';

const NotesTextarea = ({ nodeId }: { nodeId: string }) => {
  const dispatch = useAppDispatch();
  const data = useNodeData(nodeId);
  const handleNotesChanged = useCallback(
    (e: ChangeEvent<HTMLTextAreaElement>) => {
      dispatch(nodeNotesChanged({ nodeId, notes: e.target.value }));
    },
    [dispatch, nodeId]
  );
  if (!isInvocationNodeData(data)) {
    return null;
  }
  return (
    <FormControl>
      <FormLabel>Notes</FormLabel>
      <IAITextarea
        value={data?.notes}
        onChange={handleNotesChanged}
        rows={10}
      />
    </FormControl>
  );
};

export default memo(NotesTextarea);
