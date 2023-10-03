import { Box, Text } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAITextarea from 'common/components/IAITextarea';
import { workflowNotesChanged } from 'features/nodes/store/nodesSlice';
import { ChangeEvent, memo, useCallback } from 'react';

const selector = createSelector(stateSelector, ({ nodes }) => {
  const { notes } = nodes.workflow;

  return {
    notes,
  };
});

const WorkflowNotesTab = () => {
  const { notes } = useAppSelector(selector);
  const dispatch = useAppDispatch();

  const handleChangeNotes = useCallback(
    (e: ChangeEvent<HTMLTextAreaElement>) => {
      dispatch(workflowNotesChanged(e.target.value));
    },
    [dispatch]
  );

  return (
    <Box sx={{ pos: 'relative', h: 'full' }}>
      <IAITextarea
        onChange={handleChangeNotes}
        value={notes}
        fontSize="sm"
        sx={{ h: 'full', resize: 'none' }}
      />
      <Box sx={{ pos: 'absolute', bottom: 2, right: 2 }}>
        <Text
          sx={{
            fontSize: 'xs',
            opacity: 0.5,
            userSelect: 'none',
          }}
        >
          {notes.length}
        </Text>
      </Box>
    </Box>
  );
};

export default memo(WorkflowNotesTab);
