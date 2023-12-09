import { Flex, FormControl, FormLabel } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIInput from 'common/components/IAIInput';
import IAITextarea from 'common/components/IAITextarea';
import ScrollableContent from 'features/nodes/components/sidePanel/ScrollableContent';
import {
  workflowAuthorChanged,
  workflowContactChanged,
  workflowDescriptionChanged,
  workflowNameChanged,
  workflowNotesChanged,
  workflowTagsChanged,
  workflowVersionChanged,
} from 'features/nodes/store/workflowSlice';
import { ChangeEvent, memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector(stateSelector, ({ workflow }) => {
  const { author, name, description, tags, version, contact, notes } = workflow;

  return {
    name,
    author,
    description,
    tags,
    version,
    contact,
    notes,
  };
});

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

  const { t } = useTranslation();

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
          <IAIInput
            label={t('nodes.workflowName')}
            value={name}
            onChange={handleChangeName}
          />
          <IAIInput
            label={t('nodes.workflowVersion')}
            value={version}
            onChange={handleChangeVersion}
          />
        </Flex>
        <Flex sx={{ gap: 2, w: 'full' }}>
          <IAIInput
            label={t('nodes.workflowAuthor')}
            value={author}
            onChange={handleChangeAuthor}
          />
          <IAIInput
            label={t('nodes.workflowContact')}
            value={contact}
            onChange={handleChangeContact}
          />
        </Flex>
        <IAIInput
          label={t('nodes.workflowTags')}
          value={tags}
          onChange={handleChangeTags}
        />
        <FormControl as={Flex} sx={{ flexDir: 'column' }}>
          <FormLabel>{t('nodes.workflowDescription')}</FormLabel>
          <IAITextarea
            onChange={handleChangeDescription}
            value={description}
            fontSize="sm"
            sx={{ resize: 'none' }}
          />
        </FormControl>
        <FormControl as={Flex} sx={{ flexDir: 'column', h: 'full' }}>
          <FormLabel>{t('nodes.workflowNotes')}</FormLabel>
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
