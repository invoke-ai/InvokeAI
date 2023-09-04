import { Flex, FormControl, FormLabel } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIInput from 'common/components/IAIInput';
import IAITextarea from 'common/components/IAITextarea';
import {
  workflowAuthorChanged,
  workflowContactChanged,
  workflowDescriptionChanged,
  workflowNameChanged,
  workflowNotesChanged,
  workflowTagsChanged,
  workflowVersionChanged,
} from 'features/nodes/store/nodesSlice';
import { ChangeEvent, memo, useCallback } from 'react';
import ScrollableContent from '../ScrollableContent';

const selector = createSelector(
  stateSelector,
  ({ nodes }) => {
    const { author, name, description, tags, version, contact, notes } =
      nodes.workflow;

    return {
      name,
      author,
      description,
      tags,
      version,
      contact,
      notes,
    };
  },
  defaultSelectorOptions
);

const WorkflowGeneralTab = () => {
  const { author, name, description, tags, version, contact, notes } =
    useAppSelector(selector);
  const dispatch = useAppDispatch();

  const handleChangeName = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(workflowNameChanged(e.target.value));
    },
    [dispatch]
  );
  const handleChangeAuthor = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(workflowAuthorChanged(e.target.value));
    },
    [dispatch]
  );
  const handleChangeContact = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(workflowContactChanged(e.target.value));
    },
    [dispatch]
  );
  const handleChangeVersion = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(workflowVersionChanged(e.target.value));
    },
    [dispatch]
  );
  const handleChangeDescription = useCallback(
    (e: ChangeEvent<HTMLTextAreaElement>) => {
      dispatch(workflowDescriptionChanged(e.target.value));
    },
    [dispatch]
  );
  const handleChangeTags = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(workflowTagsChanged(e.target.value));
    },
    [dispatch]
  );

  const handleChangeNotes = useCallback(
    (e: ChangeEvent<HTMLTextAreaElement>) => {
      dispatch(workflowNotesChanged(e.target.value));
    },
    [dispatch]
  );

  return (
    <ScrollableContent>
      <Flex
        sx={{
          flexDir: 'column',
          alignItems: 'flex-start',
          gap: 2,
          h: 'full',
        }}
      >
        <Flex sx={{ gap: 2, w: 'full' }}>
          <IAIInput label="Name" value={name} onChange={handleChangeName} />
          <IAIInput
            label="Version"
            value={version}
            onChange={handleChangeVersion}
          />
        </Flex>
        <Flex sx={{ gap: 2, w: 'full' }}>
          <IAIInput
            label="Author"
            value={author}
            onChange={handleChangeAuthor}
          />
          <IAIInput
            label="Contact"
            value={contact}
            onChange={handleChangeContact}
          />
        </Flex>
        <IAIInput label="Tags" value={tags} onChange={handleChangeTags} />
        <FormControl as={Flex} sx={{ flexDir: 'column' }}>
          <FormLabel>Short Description</FormLabel>
          <IAITextarea
            onChange={handleChangeDescription}
            value={description}
            fontSize="sm"
            sx={{ resize: 'none' }}
          />
        </FormControl>
        <FormControl as={Flex} sx={{ flexDir: 'column', h: 'full' }}>
          <FormLabel>Notes</FormLabel>
          <IAITextarea
            onChange={handleChangeNotes}
            value={notes}
            fontSize="sm"
            sx={{ h: 'full', resize: 'none' }}
          />
        </FormControl>
      </Flex>
    </ScrollableContent>
  );
};

export default memo(WorkflowGeneralTab);
