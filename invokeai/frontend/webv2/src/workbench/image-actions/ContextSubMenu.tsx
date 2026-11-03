import type { ReactNode } from 'react';

import { HStack, Icon, Menu, Portal, Stack, Text } from '@chakra-ui/react';
import { ScrollableMenuContent } from '@platform/ui/ScrollableMenuContent';
import { ChevronRightIcon, type LucideIcon } from 'lucide-react';

/** Extras on top of the shared MenuContent surface styling, for the image menus and their submenus. */
export const IMAGE_MENU_CONTENT_PROPS = {
  minW: '13rem',
  overflow: 'hidden',
} as const;
const SUBMENU_POSITIONING_PROPS = { placement: 'right-start' } as const;
/** `MenuActionItem`'s padding for a two-line row. */
const TWO_LINE_ITEM = { py: '1.5' } as const;

/** A nested menu inside an image menu, opened from a trigger row shaped like the items beside it. */
export const ContextSubMenu = ({
  children,
  detail,
  hint,
  icon,
  label,
}: {
  children: ReactNode;
  /** A short value after the label, such as an upscaler's native scale. */
  detail?: string;
  /** Second line under the label, as `MenuActionItem` shows one; top-aligns the icon. */
  hint?: string;
  icon?: LucideIcon;
  label: string;
}) => (
  <Menu.Root positioning={SUBMENU_POSITIONING_PROPS}>
    <Menu.TriggerItem {...(hint ? TWO_LINE_ITEM : undefined)}>
      <HStack alignItems={hint ? 'flex-start' : 'center'} gap={hint ? '2.5' : '2'} minW="0" w="full">
        {icon ? <Icon as={icon} boxSize="3.5" color="fg.subtle" flexShrink={0} mt={hint ? '0.5' : undefined} /> : null}
        {hint ? (
          // Sized as `MenuActionItem` sizes a two-line row, so a hinted submenu reads like the items beside it.
          <Stack flex="1" gap="0" minW="0">
            <Text truncate>{label}</Text>
            <Text color="fg.subtle" fontSize="xs">
              {hint}
            </Text>
          </Stack>
        ) : (
          <Text flex="1" fontSize="md" minW="0" truncate>
            {label}
          </Text>
        )}
        {detail ? (
          // Secondary to the label, but at the menu's secondary-text contrast: fg.subtle is too faint on dark menus.
          <Text color="fg.muted" flexShrink={0} fontFamily="mono" fontSize="sm">
            {detail}
          </Text>
        ) : null}
        <Icon alignSelf="center" as={ChevronRightIcon} boxSize="3" color="fg.subtle" flexShrink={0} />
      </HStack>
    </Menu.TriggerItem>
    <Portal>
      <Menu.Positioner>
        {/* Boards and upscalers can run long: every submenu grows to the available height, then scrolls. */}
        <ScrollableMenuContent {...IMAGE_MENU_CONTENT_PROPS}>{children}</ScrollableMenuContent>
      </Menu.Positioner>
    </Portal>
  </Menu.Root>
);
