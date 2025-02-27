import { FormControl, FormLabel, Textarea } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useInvocationNodeNotes } from 'features/nodes/hooks/useNodeNotes';
import { nodeNotesChanged } from 'features/nodes/store/nodesSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  nodeId: string;
};

export const InvocationNodeNotesTextarea = memo(({ nodeId }: Props) => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const notes = useInvocationNodeNotes(nodeId);
  const handleNotesChanged = useCallback(
    (e: ChangeEvent<HTMLTextAreaElement>) => {
      dispatch(nodeNotesChanged({ nodeId, notes: e.target.value }));
    },
    [dispatch, nodeId]
  );
  return (
    <FormControl orientation="vertical" h="full">
      <FormLabel>{t('nodes.notes')}</FormLabel>
      <Textarea value={notes} onChange={handleNotesChanged} rows={10} resize="none" variant="darkFilled" />
    </FormControl>
  );
});

InvocationNodeNotesTextarea.displayName = 'InvocationNodeNotesTextarea';
