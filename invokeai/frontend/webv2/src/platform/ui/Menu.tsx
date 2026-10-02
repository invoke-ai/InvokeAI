import type { LucideIcon } from 'lucide-react';
import type { ComponentProps } from 'react';

import { Box, HStack, Icon, Kbd, Menu, Stack, Text, useMenuContext } from '@chakra-ui/react';

import { Tooltip } from './Tooltip';
import { useRegisterWidgetOverlay } from './widgetOverlays';

type MenuContentProps = ComponentProps<typeof Menu.Content>;

/** Menu content closes with its owning widget; the theme recipe owns chrome. */
export const MenuContent = (props: MenuContentProps) => {
  const menu = useMenuContext();
  const stale = useRegisterWidgetOverlay(menu.open, menu.setOpen);
  return stale ? null : <Menu.Content {...props} />;
};

export interface MenuActionItemProps {
  value: string;
  label: string;
  /** Second line under the label, for choices whose consequences are not obvious; top-aligns the icon. */
  hint?: string;
  icon?: LucideIcon;
  /** CSS color for the icon (e.g. a swatch); the theme tone otherwise. */
  iconColor?: string;
  tone?: 'danger';
  disabled?: boolean;
  /** Trailing keycap strings, already formatted for the platform. */
  hintParts?: readonly string[];
  onSelect: () => void;
}

const TWO_LINE_ITEM = { py: '1.5' } as const;

/** The shared icon+label menu item; `tone: 'danger'` colors the whole row, icon included. */
export const MenuActionItem = ({
  disabled,
  hint,
  hintParts,
  icon,
  iconColor,
  label,
  onSelect,
  tone,
  value,
}: MenuActionItemProps) => (
  <Menu.Item
    {...(hint ? TWO_LINE_ITEM : undefined)}
    data-danger={tone === 'danger' ? '' : undefined}
    disabled={disabled}
    value={value}
    onSelect={onSelect}
  >
    <HStack alignItems={hint ? 'flex-start' : 'center'} gap={hint ? '2.5' : '2'} minW="0" w="full">
      {icon ? (
        <Icon
          as={icon}
          boxSize="3.5"
          color={tone === 'danger' ? undefined : (iconColor ?? 'fg.subtle')}
          fill={iconColor ?? 'none'}
          flexShrink={0}
          mt={hint ? '0.5' : undefined}
        />
      ) : null}
      {hint ? (
        <Stack flex="1" gap="0" minW="0">
          <Text fontSize="xs">{label}</Text>
          <Text color="fg.subtle" fontSize="2xs">
            {hint}
          </Text>
        </Stack>
      ) : (
        <Text flex="1" fontSize="xs">
          {label}
        </Text>
      )}
      {hintParts && hintParts.length > 0 ? (
        <HStack flexShrink={0} gap="0.5">
          {hintParts.map((part) => (
            <Kbd key={part} size="sm" textTransform="lowercase">
              {part}
            </Kbd>
          ))}
        </HStack>
      ) : null}
    </HStack>
  </Menu.Item>
);

const ICON_ITEM_TOOLTIP_CONTENT_PROPS = { fontSize: '2xs' } as const;
const ICON_ITEM_TOOLTIP_POSITIONING_PROPS = { placement: 'top' } as const;

export interface MenuIconItemProps {
  value: string;
  /** The accessible name; also the hover tooltip, since the item shows only its icon. */
  label: string;
  icon: LucideIcon;
  /** Lucide icons are stroke-only, so `'currentColor'` is how an on state reads. */
  iconFill?: string;
  tone?: 'danger';
  disabled?: boolean;
  onSelect: () => void;
}

/** Wrap the icon, not Menu.Item: tooltip IDs would replace Zag's selection ID and prevent onSelect. */
export const MenuIconItem = ({ disabled, icon, iconFill, label, onSelect, tone, value }: MenuIconItemProps) => (
  <Menu.Item
    aria-label={label}
    data-danger={tone === 'danger' ? '' : undefined}
    disabled={disabled}
    flex="1"
    justifyContent="center"
    value={value}
    onSelect={onSelect}
  >
    <Tooltip
      showArrow
      content={label}
      contentProps={ICON_ITEM_TOOLTIP_CONTENT_PROPS}
      openDelay={300}
      positioning={ICON_ITEM_TOOLTIP_POSITIONING_PROPS}
    >
      <Box alignItems="center" display="flex" h="full" justifyContent="center" w="full">
        <Icon as={icon} boxSize="4" color={tone === 'danger' ? undefined : 'fg'} fill={iconFill ?? 'none'} />
      </Box>
    </Tooltip>
  </Menu.Item>
);
