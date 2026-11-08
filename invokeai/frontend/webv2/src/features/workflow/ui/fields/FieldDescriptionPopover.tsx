import type { ChangeEvent } from 'react';

import { Icon, Popover, Portal, Stack, Text, Textarea, IconButton } from '@chakra-ui/react';
import { useProjectGraphCommands } from '@features/workflow/ui/useProjectGraphCommands';
import { PopoverContent } from '@platform/ui/Popover';
import { Tooltip, useTooltipTriggerIds } from '@platform/ui/Tooltip';
import { FileTextIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const POPOVER_POSITIONING = { placement: 'bottom-end' } as const;

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
  const { t } = useTranslation();
  const triggerIds = useTooltipTriggerIds();
  const editLabel = t('nodes.editFieldDescription');
  const { editGraph } = useProjectGraphCommands();
  const onDescriptionChange = useCallback(
    (event: ChangeEvent<HTMLTextAreaElement>) =>
      editGraph({
        description: event.currentTarget.value,
        fieldName,
        nodeId,
        type: 'setFieldDescription',
      }),
    [editGraph, fieldName, nodeId]
  );

  return (
    <Popover.Root ids={triggerIds} lazyMount positioning={POPOVER_POSITIONING}>
      <Tooltip content={editLabel} ids={triggerIds}>
        <Popover.Trigger asChild>
          <IconButton
            aria-label={editLabel}
            className="nodrag"
            color={description ? 'accent.solid' : 'fg.subtle'}
            size="sm"
            variant="ghost"
          >
            <Icon as={FileTextIcon} boxSize="3" />
          </IconButton>
        </Popover.Trigger>
      </Tooltip>
      <Portal>
        <Popover.Positioner>
          <PopoverContent w="18rem">
            <Popover.Body p="2">
              <Stack gap="1">
                <Text color="fg.subtle" fontSize="xs" lineHeight={1.3}>
                  {t('nodes.fieldDescriptionHint')}
                </Text>
                <Textarea
                  aria-label={t('nodes.fieldDescription')}
                  className="nodrag nowheel"
                  fontSize="xs"
                  minH="4rem"
                  placeholder={templateDescription || t('nodes.fieldDescriptionPlaceholder')}
                  resize="vertical"
                  value={description ?? ''}
                  onChange={onDescriptionChange}
                />
              </Stack>
            </Popover.Body>
          </PopoverContent>
        </Popover.Positioner>
      </Portal>
    </Popover.Root>
  );
};
