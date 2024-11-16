import { FormControl, FormLabel, Textarea } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useNode } from 'features/nodes/hooks/useNode';
import { nodeNotesChanged } from 'features/nodes/store/nodesSlice';
import { isInvocationNode } from 'features/nodes/types/invocation';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const NotesTextarea = ({ nodeId }: { nodeId: string }) => {
  const dispatch = useAppDispatch();
  const node = useNode(nodeId);
  const { t } = useTranslation();
  const handleNotesChanged = useCallback(
    (e: ChangeEvent<HTMLTextAreaElement>) => {
      dispatch(nodeNotesChanged({ nodeId, notes: e.target.value }));
    },
    [dispatch, nodeId]
  );
  if (!isInvocationNode(node)) {
    return null;
  }
  return (
    <FormControl orientation="vertical" h="full">
      <FormLabel>{t('nodes.notes')}</FormLabel>
      <Textarea value={node.data?.notes} onChange={handleNotesChanged} rows={10} resize="none" />
    </FormControl>
  );
};

export default memo(NotesTextarea);
