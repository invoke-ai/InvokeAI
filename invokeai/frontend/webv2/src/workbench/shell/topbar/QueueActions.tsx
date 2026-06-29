import { Group, Menu, Portal } from '@chakra-ui/react';
import { IconButton, Tooltip } from '@workbench/components/ui';
import { QueueMenuItems, useQueueMenuActions } from '@workbench/widgets/queue/queueMenuActions';
import { ChevronDownIcon, XIcon } from 'lucide-react';

/** Queue cancel cluster and processor actions. */
export const QueueActions = () => {
  const actions = useQueueMenuActions();
  const cancelCurrent = actions[0];

  return (
    <Menu.Root>
      <Group attached>
        <Tooltip content="Cancel current item" showArrow>
          <IconButton
            aria-label="Cancel current queue item"
            disabled={cancelCurrent?.disabled}
            variant="outline"
            size="sm"
            roundedEnd="none"
            borderColor="border.subtle"
            onClick={cancelCurrent?.onClick}
          >
            <XIcon />
          </IconButton>
        </Tooltip>
        <Menu.Trigger asChild>
          <IconButton
            aria-label="Queue actions"
            variant="outline"
            size="sm"
            roundedStart="none"
            borderStartWidth="0"
            aspectRatio="unset"
            minW="0"
            w="6"
            borderColor="border.subtle"
          >
            <ChevronDownIcon />
          </IconButton>
        </Menu.Trigger>
      </Group>
      <Portal>
        <Menu.Positioner>
          <Menu.Content>
            <QueueMenuItems actions={actions} />
          </Menu.Content>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};
