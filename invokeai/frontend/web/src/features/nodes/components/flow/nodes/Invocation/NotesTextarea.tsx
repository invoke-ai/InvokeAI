import { FormControl, FormLabel, Flex } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import IAITextarea from 'common/components/IAITextarea';
import { useNodeNotes } from 'features/nodes/hooks/useNodeNotes';
import { nodeNotesChanged } from 'features/nodes/store/nodesSlice';
import { isNil } from 'lodash-es';
import { ChangeEvent, memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const NotesTextarea = ({ nodeId }: { nodeId: string }) => {
  const dispatch = useAppDispatch();
  const notes = useNodeNotes(nodeId);
  const { t } = useTranslation();
  const handleNotesChanged = useCallback(
    (e: ChangeEvent<HTMLTextAreaElement>) => {
      dispatch(nodeNotesChanged({ nodeId, notes: e.target.value }));
    },
    [dispatch, nodeId]
  );
  if (isNil(notes)) {
    return null;
  }
  return (
    <FormControl as={Flex} sx={{ flexDir: 'column', h: 'full' }}>
      <FormLabel>{t('nodes.notes')}</FormLabel>
      <IAITextarea
        value={notes}
        onChange={handleNotesChanged}
        resize="none"
        h="full"
      />
    </FormControl>
  );
};

export default memo(NotesTextarea);
