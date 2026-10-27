import type { LucideIcon } from 'lucide-react';
import type { ComponentProps, ReactNode, Ref } from 'react';

import {
  Box,
  chakra,
  HStack,
  Icon,
  Menu,
  Portal,
  Stack,
  Text,
  useMenuContext,
  useMenuItemContext,
} from '@chakra-ui/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { createExternalStore } from '@platform/state/externalStore';
import { useCallback, useId, useMemo } from 'react';

import { IconButton, useUnavailableControl } from './Button';
import { MiddleTruncate } from './MiddleTruncate';
import { Tooltip } from './Tooltip';
import { useRegisterWidgetOverlay } from './widgetOverlays';

type MenuContentProps = ComponentProps<typeof Menu.Content>;

/** Menu content closes with its owning widget; the theme recipe owns chrome. */
export const MenuContent = (props: MenuContentProps) => {
  const menu = useMenuContext();
  const stale = useRegisterWidgetOverlay(menu.open, menu.setOpen);
  return stale ? null : <Menu.Content {...props} />;
};

export interface MenuHeadingGroupProps {
  /** What the menu acts on, often a user's name: kept in its own case and middle-truncated. */
  label: string;
  /** Plain state text on the heading's row, such as "Unsaved changes". */
  trailing?: ReactNode;
  children: ReactNode;
}

/**
 * A menu's title over the items it names. It keeps the group label's size, weight, padding and height so a titled
 * menu reads like every other one, and labels the group, so assistive technology hears it without a focusable row.
 */
export const MenuHeadingGroup = ({ children, label, trailing }: MenuHeadingGroupProps) => (
  <Menu.ItemGroup>
    <Menu.ItemGroupLabel
      alignItems="center"
      color="fg.muted"
      display="flex"
      gap="2"
      letterSpacing="normal"
      textTransform="none"
    >
      <MiddleTruncate flex="1" text={label} />
      {trailing ? (
        <chakra.span color="fg.subtle" flexShrink={0} fontWeight="normal">
          {trailing}
        </chakra.span>
      ) : null}
    </Menu.ItemGroupLabel>
    {children}
  </Menu.ItemGroup>
);

const UNAVAILABLE_REASON_POSITIONING = { placement: 'right' } as const;

/**
 * The reason an unavailable item cannot act, shown while the item is highlighted, by pointer or keyboard alike: Zag
 * highlights with `aria-activedescendant`, so the item never takes focus and a focus-opened tooltip would never show.
 * The tooltip anchors to an overlay of the whole item, so the item's own children and layout stay as the caller wrote
 * them, and the item keeps its Zag id (a tooltip trigger would take it over).
 */
const UnavailableReason = ({ label, reason }: { label: string | undefined; reason: string }) => {
  const item = useMenuItemContext();
  const content = useMemo(
    () =>
      label ? (
        <Stack gap="0">
          <Text>{label}</Text>
          <Text color="fg.muted">{reason}</Text>
        </Stack>
      ) : (
        reason
      ),
    [label, reason]
  );
  return (
    <Tooltip content={content} open={item.highlighted} positioning={UNAVAILABLE_REASON_POSITIONING}>
      <Box aria-hidden inset="0" pointerEvents="none" position="absolute" />
    </Tooltip>
  );
};

export type MenuItemProps = ComponentProps<typeof Menu.Item> & {
  /**
   * Why the item cannot act now (say, no clipboard on a plain-HTTP origin). It reads as disabled (`aria-disabled`,
   * described by the reason) but stays highlightable, where Zag's `disabled` would skip it in keyboard navigation; no
   * selection gets through (neither its handlers nor the menu root's `onSelect`, and the menu stays open); and the
   * reason shows beside it while highlighted. An item named only by `aria-label` shows that name with the reason. The
   * reason wins over `disabled`, which would make it unreachable.
   */
  unavailableReason?: string;
};

/** `Menu.Item`, plus the unavailable state. */
export const MenuItem = ({ children, disabled, unavailableReason, ...props }: MenuItemProps) => {
  const unavailable = useUnavailableControl(unavailableReason, props['aria-describedby']);

  return (
    <Menu.Item {...props} {...unavailable?.props} disabled={unavailable ? undefined : disabled}>
      {children}
      {unavailableReason ? <UnavailableReason label={props['aria-label']} reason={unavailableReason} /> : null}
      {unavailable?.description}
    </Menu.Item>
  );
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
  /** See `MenuItem`. */
  unavailableReason?: string;
  /** Trailing keycaps; the caller renders them, since key glyphs are a workbench concern. */
  shortcut?: ReactNode;
  onSelect: () => void;
}

const TWO_LINE_ITEM = { py: '1.5' } as const;

/** The shared icon+label menu item; `tone: 'danger'` colors the whole row, icon included. */
export const MenuActionItem = ({
  disabled,
  hint,
  icon,
  iconColor,
  label,
  onSelect,
  shortcut,
  tone,
  unavailableReason,
  value,
}: MenuActionItemProps) => {
  const content = (
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
          <Text>{label}</Text>
          <Text color="fg.subtle" fontSize="xs">
            {hint}
          </Text>
        </Stack>
      ) : (
        <Text flex="1" fontSize="md">
          {label}
        </Text>
      )}
      {shortcut}
    </HStack>
  );

  return (
    <MenuItem
      {...(hint ? TWO_LINE_ITEM : undefined)}
      data-danger={tone === 'danger' ? '' : undefined}
      disabled={disabled}
      unavailableReason={unavailableReason}
      value={value}
      onSelect={onSelect}
    >
      {content}
    </MenuItem>
  );
};

const ICON_ITEM_TOOLTIP_CONTENT_PROPS = { fontSize: 'xs' } as const;
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
  /** See `MenuItem`; the reason's tooltip names the item too. */
  unavailableReason?: string;
  onSelect: () => void;
}

/** Wrap the icon, not Menu.Item: tooltip IDs would replace Zag's selection ID and prevent onSelect. */
export const MenuIconItem = ({
  disabled,
  icon,
  iconFill,
  label,
  onSelect,
  tone,
  unavailableReason,
  value,
}: MenuIconItemProps) => {
  const glyph = (
    <Box alignItems="center" display="flex" h="full" justifyContent="center" w="full">
      <Icon as={icon} boxSize="4" color={tone === 'danger' ? undefined : 'fg'} fill={iconFill ?? 'none'} />
    </Box>
  );

  return (
    <MenuItem
      aria-label={label}
      data-danger={tone === 'danger' ? '' : undefined}
      disabled={disabled}
      flex="1"
      justifyContent="center"
      unavailableReason={unavailableReason}
      value={value}
      onSelect={onSelect}
    >
      {/* The reason's tooltip names an unavailable item, so the hover label would only repeat it. */}
      {unavailableReason ? (
        glyph
      ) : (
        <Tooltip
          showArrow
          content={label}
          contentProps={ICON_ITEM_TOOLTIP_CONTENT_PROPS}
          openDelay={300}
          positioning={ICON_ITEM_TOOLTIP_POSITIONING_PROPS}
        >
          {glyph}
        </Tooltip>
      )}
    </MenuItem>
  );
};

export interface IconRadioMenuOption<Value extends string> {
  value: Value;
  label: string;
  /** Also the trigger's icon while this option is chosen. */
  icon: LucideIcon;
}

export interface IconRadioMenuProps<Value extends string> {
  /** What is being chosen; heads the menu's options. */
  label: string;
  /** The trigger's accessible name; name the current choice, since the trigger shows only its icon. */
  triggerLabel: string;
  tooltip: ReactNode;
  options: readonly IconRadioMenuOption<Value>[];
  value: Value;
  /** A choice that narrows what the surface shows reads as an active control. */
  isActive?: boolean;
  size?: 'sm' | 'md';
  /** A stable id lets another surface hand focus to the trigger; a generated one otherwise. */
  triggerId?: string;
  triggerRef?: Ref<HTMLButtonElement>;
  triggerAttributes?: Readonly<Record<`data-${string}`, string>>;
  onValueChange: (value: Value) => void;
}

const ICON_RADIO_MENU_POSITIONING = { placement: 'bottom-end' } as const;

/** A compact single-choice menu behind an icon trigger that shows the current choice. */
export const IconRadioMenu = <Value extends string>({
  isActive = false,
  label,
  onValueChange,
  options,
  size = 'sm',
  tooltip,
  triggerAttributes,
  triggerId,
  triggerLabel,
  triggerRef,
  value,
}: IconRadioMenuProps<Value>) => {
  const generatedId = useId();
  // Shared ids let the tooltip ride the trigger without wrapping Menu.Trigger, which would swallow its anchor ref.
  const ids = useMemo(() => ({ trigger: triggerId ?? generatedId }), [generatedId, triggerId]);
  const current = options.find((option) => option.value === value) ?? options[0];
  const handleValueChange = useCallback(
    (event: { value: string }) => {
      const next = options.find((option) => option.value === event.value);

      if (next && next.value !== value) {
        onValueChange(next.value);
      }
    },
    [onValueChange, options, value]
  );

  return (
    <Menu.Root ids={ids} positioning={ICON_RADIO_MENU_POSITIONING}>
      <Tooltip content={tooltip} ids={ids}>
        <Menu.Trigger asChild>
          <IconButton
            ref={triggerRef}
            aria-label={triggerLabel}
            color={isActive ? undefined : 'fg.muted'}
            data-active={isActive ? '' : undefined}
            flexShrink={0}
            size={size}
            variant={isActive ? 'solid' : 'ghost'}
            {...triggerAttributes}
          >
            {current ? <Icon as={current.icon} boxSize="3.5" /> : null}
          </IconButton>
        </Menu.Trigger>
      </Tooltip>
      <Portal>
        <Menu.Positioner>
          <MenuContent minW="10rem">
            <Menu.ItemGroup>
              <Menu.ItemGroupLabel>{label}</Menu.ItemGroupLabel>
              <Menu.RadioItemGroup value={value} onValueChange={handleValueChange}>
                {options.map((option) => (
                  <Menu.RadioItem key={option.value} value={option.value}>
                    <Menu.ItemIndicator />
                    <Menu.ItemText>{option.label}</Menu.ItemText>
                  </Menu.RadioItem>
                ))}
              </Menu.RadioItemGroup>
            </Menu.ItemGroup>
          </MenuContent>
        </Menu.Positioner>
      </Portal>
    </Menu.Root>
  );
};

interface ContextMenuAnchor {
  x: number;
  y: number;
  focusTarget?: () => HTMLElement | null;
  restoreFocus?: () => void;
}

// Only the active gesture is retained. Actions and their enabled state belong to the live menu's render.
const activeMenu = createExternalStore<{ owner: string | null; anchor: ContextMenuAnchor | null }>({
  owner: null,
  anchor: null,
});

/**
 * Menus sharing this ownership replace one another; none is a parent of the next. The layer stack dismisses every
 * layer registered after one being torn down, so a menu that registers before its predecessor is gone (its content
 * still mounted from an exit animation) would be closed as if nested. Pass this as the menu root's `onRequestDismiss`.
 */
const keepOpenThroughSiblingTeardown = (event: Event) => event.preventDefault();

/** Row and field menus share ownership so a second right-click replaces the first menu. */
export const useContextMenu = () => {
  const owner = useId();
  const anchor = activeMenu.useSelector((menu) => (menu.owner === owner ? menu.anchor : null));
  const open = useCallback((anchor: ContextMenuAnchor) => activeMenu.setSnapshot({ owner, anchor }), [owner]);
  const close = useCallback(() => {
    const menu = activeMenu.getSnapshot();
    if (menu.owner !== owner) {
      return;
    }
    activeMenu.setSnapshot({ owner: null, anchor: null });
    menu.anchor?.restoreFocus?.();
  }, [owner]);

  useMountEffect(() => () => {
    if (activeMenu.getSnapshot().owner === owner) {
      activeMenu.setSnapshot({ owner: null, anchor: null });
    }
  });

  return { anchor, close, onRequestDismiss: keepOpenThroughSiblingTeardown, open };
};
