import type { WorkflowMetadata } from '@workbench/workflows/types';
import type { ChangeEvent } from 'react';

import { HStack, Input, Stack, Textarea } from '@chakra-ui/react';
import { Field } from '@workbench/components/ui';
import { useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { useCallback } from 'react';

/** Workflow metadata editing — the legacy "Details" tab. */
export const WorkflowDetailsTab = ({ metadata }: { metadata: WorkflowMetadata }) => {
  const dispatch = useWorkbenchDispatch();
  const commit = useCallback(
    (patch: Partial<WorkflowMetadata>) =>
      dispatch({ action: { patch, type: 'setMetadata' }, type: 'applyProjectGraphAction' }),
    [dispatch]
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
      {textField('Name', 'name', onNameChange)}
      <Field label="Description">
        <Textarea
          minH="3.5rem"
          resize="vertical"
          size="xs"
          value={metadata.description}
          onChange={onDescriptionChange}
        />
      </Field>
      <HStack align="start" gap="2">
        {textField('Author', 'author', onAuthorChange)}
        {textField('Version', 'workflowVersion', onWorkflowVersionChange)}
      </HStack>
      <HStack align="start" gap="2">
        {textField('Tags', 'tags', onTagsChange)}
        {textField('Contact', 'contact', onContactChange)}
      </HStack>
      <Field label="Notes">
        <Textarea minH="3.5rem" resize="vertical" size="xs" value={metadata.notes} onChange={onNotesChange} />
      </Field>
    </Stack>
  );
};
