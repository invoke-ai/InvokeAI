import { Flex } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvControlGroup } from 'common/components/InvControl/InvControlGroup';
import { InvInput } from 'common/components/InvInput/InvInput';
import { InvTextarea } from 'common/components/InvTextarea/InvTextarea';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import {
 selectWorkflowSlice,  workflowAuthorChanged,
  workflowContactChanged,
  workflowDescriptionChanged,
  workflowNameChanged,
  workflowNotesChanged,
  workflowTagsChanged,
  workflowVersionChanged } from 'features/nodes/store/workflowSlice';
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
      <Flex flexDir="column" alignItems="flex-start" gap={2} h="full">
        <InvControlGroup orientation="vertical">
          <Flex gap={2} w="full">
            <InvControl label={t('nodes.workflowName')}>
              <InvInput value={name} onChange={handleChangeName} />
            </InvControl>
            <InvControl label={t('nodes.workflowVersion')}>
              <InvInput value={version} onChange={handleChangeVersion} />
            </InvControl>
          </Flex>
          <Flex gap={2} w="full">
            <InvControl label={t('nodes.workflowAuthor')}>
              <InvInput value={author} onChange={handleChangeAuthor} />
            </InvControl>
            <InvControl label={t('nodes.workflowContact')}>
              <InvInput value={contact} onChange={handleChangeContact} />
            </InvControl>
          </Flex>
          <InvControl label={t('nodes.workflowTags')}>
            <InvInput value={tags} onChange={handleChangeTags} />
          </InvControl>
        </InvControlGroup>
        <InvControl
          label={t('nodes.workflowDescription')}
          orientation="vertical"
        >
          <InvTextarea
            onChange={handleChangeDescription}
            value={description}
            fontSize="sm"
            resize="none"
            rows={3}
          />
        </InvControl>
        <InvControl label={t('nodes.workflowNotes')} orientation="vertical">
          <InvTextarea
            onChange={handleChangeNotes}
            value={notes}
            fontSize="sm"
            resize="none"
            rows={10}
          />
        </InvControl>
      </Flex>
    </ScrollableContent>
  );
};

export default memo(WorkflowGeneralTab);
