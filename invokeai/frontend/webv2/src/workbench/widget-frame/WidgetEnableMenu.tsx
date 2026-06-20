import type {
  RegisteredWidget,
  WidgetIconComponent,
  WidgetInstanceId,
  WidgetRegion,
  WidgetTypeId,
} from '@workbench/types';

import { Flex, Icon, Menu, Portal, Text } from '@chakra-ui/react';
import { IconButton } from '@workbench/components/ui';
import { WidgetIcon } from '@workbench/iconResolver';
import { CheckIcon, MoreHorizontalIcon } from 'lucide-react';

export interface WidgetEnableMenuItem {
  allowMultiple: boolean;
  failureMessage?: string;
  icon: WidgetIconComponent;
  id: WidgetInstanceId;
  isEnabled: boolean;
  label: string;
  status?: RegisteredWidget['status'];
  typeId: WidgetTypeId;
  widget: RegisteredWidget;
}

type WidgetEnableMenuTrigger =
  | { kind: 'rail'; region: Exclude<WidgetRegion, 'bottom' | 'center'> }
  | { kind: 'bottom' }
  | { kind: 'center' };

interface WidgetEnableMenuProps {
  groupLabel: string;
  items: WidgetEnableMenuItem[];
  positioning: NonNullable<Menu.RootProps['positioning']>;
  trigger: WidgetEnableMenuTrigger;
  triggerLabel: string;
  contextTarget?: { x: number; y: number } | null;
  getItemMeta?: (item: WidgetEnableMenuItem) => string | null;
  isItemDisabled?: (item: WidgetEnableMenuItem) => boolean;
  onContextClose?: () => void;
  onToggle: (item: WidgetEnableMenuItem) => void;
}

const getWidgetEnableMenuTriggerButton = (label: string, trigger: WidgetEnableMenuTrigger) => {
  if (trigger.kind === 'center') {
    return (
      <IconButton aria-label={label} size="xs" variant="ghost">
        <Icon as={MoreHorizontalIcon} boxSize="4" />
      </IconButton>
    );
  }

  const isBottom = trigger.kind === 'bottom';

  return (
    <Flex
      align="center"
      aria-label={label}
      as="button"
      color="fg"
      h={isBottom ? '5' : '9'}
      justify="center"
      rounded={isBottom ? 'sm' : 'md'}
      transition="background var(--wb-motion-duration-fast) ease, color var(--wb-motion-duration-fast) ease"
      w={isBottom ? '5' : '9'}
      _hover={{ bg: 'bg.muted' }}
    >
      <Icon as={MoreHorizontalIcon} boxSize={isBottom ? '4' : '5'} />
    </Flex>
  );
};

export const WidgetEnableMenu = ({
  contextTarget,
  getItemMeta,
  groupLabel,
  isItemDisabled,
  items,
  onContextClose,
  onToggle,
  positioning,
  trigger,
  triggerLabel,
}: WidgetEnableMenuProps) => {
  const content = (
    <Menu.Content minW="12rem">
      <Menu.ItemGroup>
        <Menu.ItemGroupLabel color="fg.subtle" fontSize="2xs" textTransform="uppercase">
          {groupLabel}
        </Menu.ItemGroupLabel>
        {items.map((item) => {
          const meta = getItemMeta?.(item) ?? (item.failureMessage ? 'Failed' : null);
          const disabled = item.status === 'disabled' || isItemDisabled?.(item) === true;

          return (
            <Menu.Item
              key={item.id}
              role="menuitemcheckbox"
              aria-checked={item.isEnabled}
              value={item.id}
              closeOnSelect={false}
              disabled={disabled}
              _disabled={{ opacity: 0.4 }}
              onClick={() => onToggle(item)}
            >
              <Icon as={CheckIcon} boxSize="3" opacity={item.isEnabled ? 1 : 0} />
              <WidgetIcon icon={item.icon} boxSize="3.5" />
              <Menu.ItemText>{item.label}</Menu.ItemText>
              {meta ? (
                <Text color="fg.subtle" fontSize="2xs" ms="auto">
                  {meta}
                </Text>
              ) : null}
            </Menu.Item>
          );
        })}
      </Menu.ItemGroup>
    </Menu.Content>
  );

  return (
    <>
      <Menu.Root positioning={positioning}>
        <Menu.Trigger asChild>{getWidgetEnableMenuTriggerButton(triggerLabel, trigger)}</Menu.Trigger>
        <Portal>
          <Menu.Positioner>{content}</Menu.Positioner>
        </Portal>
      </Menu.Root>
      <Menu.Root
        lazyMount
        open={contextTarget !== null && contextTarget !== undefined}
        positioning={{
          ...positioning,
          getAnchorRect: () => (contextTarget ? { height: 1, width: 1, x: contextTarget.x, y: contextTarget.y } : null),
        }}
        unmountOnExit
        onOpenChange={(event) => {
          if (!event.open) {
            onContextClose?.();
          }
        }}
      >
        <Portal>
          <Menu.Positioner>{content}</Menu.Positioner>
        </Portal>
      </Menu.Root>
    </>
  );
};
