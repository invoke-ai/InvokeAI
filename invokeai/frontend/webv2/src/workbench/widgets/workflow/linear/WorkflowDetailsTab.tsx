import { HStack, Input, Stack, Textarea } from '@chakra-ui/react';
import type { ChangeEvent } from 'react';

import { Field } from '../../../components/ui/Field';
import { useWorkbench } from '../../../WorkbenchContext';
import type { WorkflowMetadata } from '../../../workflows/types';

/** Workflow metadata editing — the legacy "Details" tab. */
export const WorkflowDetailsTab = ({ metadata }: { metadata: WorkflowMetadata }) => {
  const { dispatch } = useWorkbench();
  const commit = (patch: Partial<WorkflowMetadata>) =>
    dispatch({ action: { patch, type: 'setMetadata' }, type: 'applyProjectGraphAction' });

  const textField = (label: string, key: keyof WorkflowMetadata) => (
    <Field label={label}>
      <Input
        size="xs"
        value={metadata[key]}
        onChange={(event: ChangeEvent<HTMLInputElement>) => commit({ [key]: event.currentTarget.value })}
      />
    </Field>
  );

  return (
    <Stack gap="3" p="3">
      {textField('Name', 'name')}
      <Field label="Description">
        <Textarea
          minH="3.5rem"
          resize="vertical"
          size="xs"
          value={metadata.description}
          onChange={(event: ChangeEvent<HTMLTextAreaElement>) => commit({ description: event.currentTarget.value })}
        />
      </Field>
      <HStack align="start" gap="2">
        {textField('Author', 'author')}
        {textField('Version', 'workflowVersion')}
      </HStack>
      <HStack align="start" gap="2">
        {textField('Tags', 'tags')}
        {textField('Contact', 'contact')}
      </HStack>
      <Field label="Notes">
        <Textarea
          minH="3.5rem"
          resize="vertical"
          size="xs"
          value={metadata.notes}
          onChange={(event: ChangeEvent<HTMLTextAreaElement>) => commit({ notes: event.currentTarget.value })}
        />
      </Field>
    </Stack>
  );
};
