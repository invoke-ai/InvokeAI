import { Icon, Popover, Portal, Stack, Text, Textarea } from '@chakra-ui/react';
import { FileTextIcon } from 'lucide-react';
import type { ChangeEvent } from 'react';

import { IconButton } from '../../../components/ui/Button';
import { useWorkbenchDispatch } from '../../../WorkbenchContext';

/**
 * Small textarea popover overriding a field's template description ("field
 * notes"), shared by the editor's node field rows and the form builder.
 */
export const FieldDescriptionPopover = ({
  description,
  fieldName,
  nodeId,
  templateDescription,
}: {
  /** The current override, if any. */
  description: string | undefined;
  fieldName: string;
  nodeId: string;
  templateDescription: string;
}) => {
  const dispatch = useWorkbenchDispatch();

  return (
    <Popover.Root lazyMount positioning={{ placement: 'bottom-end' }}>
      <Popover.Trigger asChild>
        <IconButton
          aria-label="Edit field description"
          className="nodrag"
          color={description ? 'accent.solid' : 'fg.subtle'}
          size="2xs"
          title="Edit field description"
          variant="ghost"
        >
          <Icon as={FileTextIcon} boxSize="3" />
        </IconButton>
      </Popover.Trigger>
      <Portal>
        <Popover.Positioner>
          <Popover.Content bg="bg.muted" borderColor="border.emphasized" borderWidth="1px" w="18rem">
            <Popover.Body p="2">
              <Stack gap="1">
                <Text color="fg.subtle" fontSize="2xs">
                  Field description — overrides the node's default. Clear to restore it.
                </Text>
                <Textarea
                  aria-label="Field description"
                  className="nodrag nowheel"
                  fontSize="2xs"
                  minH="4rem"
                  placeholder={templateDescription || 'Describe this field…'}
                  resize="vertical"
                  size="xs"
                  value={description ?? ''}
                  onChange={(event: ChangeEvent<HTMLTextAreaElement>) =>
                    dispatch({
                      action: {
                        description: event.currentTarget.value,
                        fieldName,
                        nodeId,
                        type: 'setFieldDescription',
                      },
                      type: 'applyProjectGraphAction',
                    })
                  }
                />
              </Stack>
            </Popover.Body>
          </Popover.Content>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
  );
};
