import type { FormControlProps } from '@invoke-ai/ui-library';
import {
  Box,
  Checkbox,
  Flex,
  FormControl,
  FormControlGroup,
  FormLabel,
  Image,
  Input,
  Textarea,
} from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { selectCurrentUser } from 'features/auth/store/authSlice';
import {
  workflowAuthorChanged,
  workflowContactChanged,
  workflowDescriptionChanged,
  workflowNameChanged,
  workflowNotesChanged,
  workflowTagsChanged,
  workflowVersionChanged,
} from 'features/nodes/store/nodesSlice';
import {
  selectWorkflowAuthor,
  selectWorkflowContact,
  selectWorkflowDescription,
  selectWorkflowId,
  selectWorkflowName,
  selectWorkflowNotes,
  selectWorkflowTags,
  selectWorkflowVersion,
} from 'features/nodes/store/selectors';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetSetupStatusQuery } from 'services/api/endpoints/auth';
import { useGetWorkflowQuery, useUpdateWorkflowIsPublicMutation } from 'services/api/endpoints/workflows';

import { WorkflowThumbnailEditor } from './WorkflowThumbnail/WorkflowThumbnailEditor';

const WorkflowGeneralTab = () => {
  const id = useAppSelector(selectWorkflowId);
  const name = useAppSelector(selectWorkflowName);
  const description = useAppSelector(selectWorkflowDescription);
  const notes = useAppSelector(selectWorkflowNotes);
  const author = useAppSelector(selectWorkflowAuthor);
  const contact = useAppSelector(selectWorkflowContact);
  const tags = useAppSelector(selectWorkflowTags);
  const version = useAppSelector(selectWorkflowVersion);

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
          <ShareWorkflowCheckbox id={id} />
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

const ShareWorkflowCheckbox = ({ id }: { id?: string | null }) => {
  const { t } = useTranslation();
  const currentUser = useAppSelector(selectCurrentUser);
  const { data: setupStatus } = useGetSetupStatusQuery();
  const { data } = useGetWorkflowQuery(id ?? skipToken);
  const [updateIsPublic, { isLoading }] = useUpdateWorkflowIsPublicMutation();

  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      if (!id) {
        return;
      }
      updateIsPublic({ workflow_id: id, is_public: e.target.checked });
    },
    [id, updateIsPublic]
  );

  // Only show for saved user workflows in multiuser mode when the current user is the owner or admin
  if (!data || !id || data.workflow.meta.category !== 'user') {
    return null;
  }
  if (setupStatus?.multiuser_enabled) {
    const isOwner = currentUser !== null && data.user_id === currentUser.user_id;
    const isAdmin = currentUser?.is_admin ?? false;
    if (!isOwner && !isAdmin) {
      return null;
    }
  }

  return (
    <Flex alignItems="center" gap={2}>
      <Checkbox isChecked={data.is_public} onChange={handleChange} isDisabled={isLoading} />
      <FormLabel mb={0}>{t('workflows.shareWorkflow')}</FormLabel>
    </Flex>
  );
};
