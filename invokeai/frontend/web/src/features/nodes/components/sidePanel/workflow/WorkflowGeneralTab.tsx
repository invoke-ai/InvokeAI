import type { FormControlProps } from '@invoke-ai/ui-library';
import { Box, Flex, FormControl, FormControlGroup, FormLabel, Image, Input, Textarea } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
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
import { useGetWorkflowQuery } from 'services/api/endpoints/workflows';

import { WorkflowThumbnailEditor } from './WorkflowThumbnail/WorkflowThumbnailEditor';

const selector = createMemoizedSelector(selectWorkflowSlice, (workflow) => {
  const { id, author, name, description, tags, version, contact, notes } = workflow;

  return {
    id,
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
  const { id, author, name, description, tags, version, contact, notes } = useAppSelector(selector);
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
          <FormControl>
            <FormLabel>{t('nodes.workflowName')}</FormLabel>
            <Input variant="darkFilled" value={name} onChange={handleChangeName} />
          </FormControl>
          <Thumbnail id={id} />
          <FormControl>
            <FormLabel>{t('nodes.workflowVersion')}</FormLabel>
            <Input variant="darkFilled" value={version} onChange={handleChangeVersion} />
          </FormControl>
          <FormControl>
            <FormLabel>{t('nodes.workflowAuthor')}</FormLabel>
            <Input variant="darkFilled" value={author} onChange={handleChangeAuthor} />
          </FormControl>
          <FormControl>
            <FormLabel>{t('nodes.workflowContact')}</FormLabel>
            <Input variant="darkFilled" value={contact} onChange={handleChangeContact} />
          </FormControl>
          <FormControl>
            <FormLabel>{t('nodes.workflowTags')}</FormLabel>
            <Input variant="darkFilled" value={tags} onChange={handleChangeTags} />
          </FormControl>
          <FormControl>
            <FormLabel>{t('nodes.workflowDescription')}</FormLabel>
            <Textarea
              variant="darkFilled"
              onChange={handleChangeDescription}
              value={description}
              fontSize="sm"
              resize="none"
              rows={3}
            />
          </FormControl>
          <FormControl>
            <FormLabel>{t('nodes.workflowNotes')}</FormLabel>
            <Textarea
              variant="darkFilled"
              onChange={handleChangeNotes}
              value={notes}
              fontSize="sm"
              resize="none"
              rows={10}
            />
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

const Thumbnail = ({ id }: { id?: string | null }) => {
  const { t } = useTranslation();

  const { data } = useGetWorkflowQuery(id ?? skipToken);

  if (!data) {
    return null;
  }

  if (data.workflow.meta.category === 'default' && data.thumbnail_url) {
    // This is a default workflow and it has a thumbnail set. Users may only view the thumbnail.
    return (
      <FormControl>
        <FormLabel>{t('workflows.workflowThumbnail')}</FormLabel>
        <Box position="relative" flexShrink={0}>
          <Image
            src={data.thumbnail_url}
            objectFit="cover"
            objectPosition="50% 50%"
            w={100}
            h={100}
            borderRadius="base"
          />
        </Box>
      </FormControl>
    );
  }

  if (data.workflow.meta.category !== 'default') {
    // This is a user workflow and they may edit the thumbnail.
    return (
      <FormControl>
        <FormLabel>{t('workflows.workflowThumbnail')}</FormLabel>
        <WorkflowThumbnailEditor thumbnailUrl={data.thumbnail_url} workflowId={data.workflow_id} />
      </FormControl>
    );
  }

  // This is a default workflow and it does not have a thumbnail set. Users may not edit the thumbnail.
  return null;
};
