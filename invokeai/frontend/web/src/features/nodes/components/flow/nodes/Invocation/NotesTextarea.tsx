import { FormControl, FormLabel, Textarea } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useNodeData } from 'features/nodes/hooks/useNodeData';
import { nodeNotesChanged } from 'features/nodes/store/nodesSlice';
import { isInvocationNodeData } from 'features/nodes/types/invocation';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const NotesTextarea = ({ nodeId }: { nodeId: string }) => {
  const dispatch = useAppDispatch();
  const data = useNodeData(nodeId);
  const { t } = useTranslation();
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
    <FormControl orientation="vertical" h="full">
      <FormLabel>{t('nodes.notes')}</FormLabel>
      <Textarea value={data?.notes} onChange={handleNotesChanged} rows={10} resize="none" />
    </FormControl>
  );
};

export default memo(NotesTextarea);
