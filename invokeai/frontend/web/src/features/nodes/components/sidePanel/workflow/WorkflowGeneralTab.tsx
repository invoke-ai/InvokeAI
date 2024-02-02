import type { FormControlProps } from '@invoke-ai/ui-library';
import { Flex, FormControl, FormControlGroup, FormLabel, Input, Textarea } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import {
  selectWorkflowSlice,
  workflowAuthorChanged,
  workflowContactChanged,
  workflowDescriptionChanged,
  workflowNameChanged,
  workflowNotesChanged,
  workflowTagsChanged,
  workflowVersionChanged,
} from 'features/nodes/store/workflowSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector(selectWorkflowSlice, (workflow) => {
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
  const { author, name, description, tags, version, contact, notes } = useAppSelector(selector);
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
      <Flex flexDir="column" alignItems="flex-start" gap={2} h="full">
        <FormControlGroup orientation="vertical" formControlProps={formControlProps}>
          <Flex gap={2} w="full">
            <FormControl>
              <FormLabel>{t('nodes.workflowName')}</FormLabel>
              <Input value={name} onChange={handleChangeName} />
            </FormControl>
            <FormControl>
              <FormLabel>{t('nodes.workflowVersion')}</FormLabel>
              <Input value={version} onChange={handleChangeVersion} />
            </FormControl>
          </Flex>
          <Flex gap={2} w="full">
            <FormControl>
              <FormLabel>{t('nodes.workflowAuthor')}</FormLabel>
              <Input value={author} onChange={handleChangeAuthor} />
            </FormControl>
            <FormControl>
              <FormLabel>{t('nodes.workflowContact')}</FormLabel>
              <Input value={contact} onChange={handleChangeContact} />
            </FormControl>
          </Flex>
          <FormControl>
            <FormLabel>{t('nodes.workflowTags')}</FormLabel>
            <Input value={tags} onChange={handleChangeTags} />
          </FormControl>
          <FormControl>
            <FormLabel>{t('nodes.workflowDescription')}</FormLabel>
            <Textarea onChange={handleChangeDescription} value={description} fontSize="sm" resize="none" rows={3} />
          </FormControl>
          <FormControl>
            <FormLabel>{t('nodes.workflowNotes')}</FormLabel>
            <Textarea onChange={handleChangeNotes} value={notes} fontSize="sm" resize="none" rows={10} />
          </FormControl>
        </FormControlGroup>
      </Flex>
    </ScrollableContent>
  );
};

export default memo(WorkflowGeneralTab);

const formControlProps: FormControlProps = {
  flexShrink: 0,
};
