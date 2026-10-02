import type { WorkflowMetadata } from '@features/workflow/contracts';
import type { ChangeEvent } from 'react';

import { HStack, Input, Stack, Textarea } from '@chakra-ui/react';
import { useProjectGraphCommands } from '@features/workflow/ui/useProjectGraphCommands';
import { Field } from '@platform/ui';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

/** Workflow metadata editing — the legacy "Details" tab. */
export const WorkflowDetailsTab = ({ metadata }: { metadata: WorkflowMetadata }) => {
  const { t } = useTranslation();
  const { editGraph } = useProjectGraphCommands();
  const commit = useCallback(
    (patch: Partial<WorkflowMetadata>) => editGraph({ patch, type: 'setMetadata' }),
    [editGraph]
  );

  const onNameChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => commit({ name: event.currentTarget.value }),
    [commit]
  );
  const onAuthorChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => commit({ author: event.currentTarget.value }),
    [commit]
  );
  const onWorkflowVersionChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => commit({ workflowVersion: event.currentTarget.value }),
    [commit]
  );
  const onTagsChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => commit({ tags: event.currentTarget.value }),
    [commit]
  );
  const onContactChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => commit({ contact: event.currentTarget.value }),
    [commit]
  );
  const onDescriptionChange = useCallback(
    (event: ChangeEvent<HTMLTextAreaElement>) => commit({ description: event.currentTarget.value }),
    [commit]
  );
  const onNotesChange = useCallback(
    (event: ChangeEvent<HTMLTextAreaElement>) => commit({ notes: event.currentTarget.value }),
    [commit]
  );

  const textField = (
    label: string,
    key: keyof WorkflowMetadata,
    onChange: (event: ChangeEvent<HTMLInputElement>) => void
  ) => (
    <Field label={label}>
      <Input size="xs" value={metadata[key]} onChange={onChange} />
    </Field>
  );

  return (
    <Stack gap="3" p="3">
      {textField(t('common.name'), 'name', onNameChange)}
      <Field label={t('widgets.workflow.description')}>
        <Textarea
          minH="3.5rem"
          resize="vertical"
          size="xs"
          value={metadata.description}
          onChange={onDescriptionChange}
        />
      </Field>
      <HStack align="start" gap="2">
        {textField(t('widgets.workflow.author'), 'author', onAuthorChange)}
        {textField(t('widgets.workflow.version'), 'workflowVersion', onWorkflowVersionChange)}
      </HStack>
      <HStack align="start" gap="2">
        {textField(t('widgets.workflow.tags'), 'tags', onTagsChange)}
        {textField(t('widgets.workflow.contact'), 'contact', onContactChange)}
      </HStack>
      <Field label={t('widgets.workflow.notes')}>
        <Textarea minH="3.5rem" resize="vertical" size="xs" value={metadata.notes} onChange={onNotesChange} />
      </Field>
    </Stack>
  );
};
