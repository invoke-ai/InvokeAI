import type { LucideIcon } from 'lucide-react';
import type { ComponentProps, MouseEvent, ReactNode, Ref } from 'react';

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
import { createExternalStore, createExternalStoreCore } from '@platform/state/externalStore';
import { createContext, useCallback, useContext, useId, useMemo, useRef, useSyncExternalStore } from 'react';
import { useTranslation } from 'react-i18next';

import { IconButton, useUnavailableControl } from './Button';
import { MiddleTruncate } from './MiddleTruncate';
import { Tooltip } from './Tooltip';
import { useRegisterWidgetOverlay } from './widgetOverlays';

type MenuContentProps = Omit<ComponentProps<typeof Menu.Content>, 'ref'>;

/** Grows to the context menus' cap or the measured room, then the list scrolls inside the padded box. */
export const SCROLLABLE_MENU_CONTENT_PROPS = {
  display: 'flex',
  flexDirection: 'column',
  maxH: 'min(28rem, var(--available-height))',
  overflow: 'hidden',
} as const;

/** Zag's cancelable event on the content for a press outside it; cancelling it keeps the menu open. */
const POINTER_DOWN_OUTSIDE_EVENT = 'pointerdown.outside';

/** The arrow takes no focus: a press on its tip would otherwise move focus to the page and end keyboard navigation. */
const keepFocus = (event: MouseEvent<HTMLElement>) => event.preventDefault();

const InsideMenuContentContext = createContext(false);

/**
 * The arrow is the content's next sibling, not its child (content scrolls and would clip it), and must stay after it:
 * Zag reads the z-index from the positioner's first child, and the recipe hides it through a sibling selector.
 */
export const MenuContent = ({ children, ...props }: MenuContentProps) => {
  const menu = useMenuContext();
  const isSubmenu = useContext(InsideMenuContentContext);
  const stale = useRegisterWidgetOverlay(menu.open, menu.setOpen);
  const arrowRef = useRef<HTMLDivElement>(null);
  // Zag counts the sibling arrow as outside; Ark exposes no dismissal hook here, so this listens for Zag's event first.
  const keepOpenOnArrow = useCallback((content: HTMLDivElement | null) => {
    const handlePointerDownOutside = (event: Event) => {
      const target = event instanceof CustomEvent ? (event.detail as { target?: unknown } | null)?.target : null;

      if (target instanceof Node && arrowRef.current?.contains(target)) {
        event.preventDefault();
      }
    };

    content?.addEventListener(POINTER_DOWN_OUTSIDE_EVENT, handlePointerDownOutside);
    return () => content?.removeEventListener(POINTER_DOWN_OUTSIDE_EVENT, handlePointerDownOutside);
  }, []);
  if (stale) {
    return null;
  }
  // Zag puts the placement on the content's props; their React type lists only standard attributes.
  const { 'data-placement': placement } = menu.getContentProps() as { 'data-placement'?: string };
  return (
    <>
      <Menu.Content ref={keepOpenOnArrow} {...props}>
        <InsideMenuContentContext value>{children}</InsideMenuContentContext>
      </Menu.Content>
      {isSubmenu ? null : (
        <Menu.Arrow
          ref={arrowRef}
          data-placement={placement}
          data-state={menu.open ? 'open' : 'closed'}
          onMouseDown={keepFocus}
        >
          <Menu.ArrowTip />
        </Menu.Arrow>
      )}
    </>
  );
};

export interface MenuHeadingGroupProps {
  /** What the menu acts on, often a user's name: kept in its own case and middle-truncated. */
  label: string;
  trailing?: ReactNode;
  children: ReactNode;
}

/** Labels the group, so assistive technology hears the title without a focusable row. */
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

/** Zag highlights via `aria-activedescendant`, so the item never takes focus; the tooltip anchors to an overlay. */
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
  /** Stays highlightable, unlike Zag's `disabled`, but lets no selection through; wins over `disabled`. */
  unavailableReason?: string;
};

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
  /** Describes the item (`aria-describedby`); the item's name stays the label. */
  hint?: string;
  icon?: LucideIcon;
  iconColor?: string;
  tone?: 'danger';
  disabled?: boolean;
  unavailableReason?: string;
  /** Trailing keycaps; the caller renders them, since key glyphs are a workbench concern. */
  shortcut?: ReactNode;
  onSelect: () => void;
}

const TWO_LINE_ITEM = { py: '1.5' } as const;

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
  const labelId = useId();
  const hintId = useId();
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
          <Text id={labelId}>{label}</Text>
          <Text id={hintId} color="fg.subtle" fontSize="xs">
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
      aria-describedby={hint ? hintId : undefined}
      aria-labelledby={hint ? labelId : undefined}
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

export type MenuPartSnapshot<Part> = { part: Part; status: 'loaded' } | { status: 'failed' | 'idle' | 'loading' };

export interface MenuPartLoader<Part> {
  getSnapshot: () => MenuPartSnapshot<Part>;
  /** Starts the import unless it has loaded or is under way; after a failure, imports again. Never throws or rejects. */
  load: () => void;
  subscribe: (listener: () => void) => () => void;
}

/** State, not `lazy()`: a lazy boundary suspends on first render, so the menu would open on a placeholder. */
export const createMenuPartLoader = <Part,>(importPart: () => Promise<Part>): MenuPartLoader<Part> => {
  const store = createExternalStoreCore<MenuPartSnapshot<Part>>({ status: 'idle' });

  return {
    getSnapshot: store.getSnapshot,
    load: () => {
      const { status } = store.getSnapshot();

      if (status === 'idle' || status === 'failed') {
        store.setSnapshot({ status: 'loading' });
        // An importer that throws at once fails like one that rejects, rather than leaving the part loading for good.
        new Promise<Part>((resolve) => {
          resolve(importPart());
        }).then(
          (part) => store.setSnapshot({ part, status: 'loaded' }),
          () => store.setSnapshot({ status: 'failed' })
        );
      }
    },
    subscribe: store.subscribe,
  };
};

export const useMenuPart = <Part,>(loader: MenuPartLoader<Part>): MenuPartSnapshot<Part> =>
  useSyncExternalStore(loader.subscribe, loader.getSnapshot, loader.getSnapshot);

const ignoreSelect = () => undefined;

/** Not `disabled`: Zag skips disabled items, so a menu holding only this would leave the keyboard nothing to reach. */
export const MenuPartPlaceholder = ({ status }: { status: MenuPartSnapshot<unknown>['status'] }) => {
  const { t } = useTranslation();
  const failed = status === 'failed';

  return (
    <MenuActionItem
      label={t(failed ? 'common.menuPartLoadFailed' : 'common.loading')}
      unavailableReason={t(failed ? 'common.menuPartLoadFailedReason' : 'common.menuPartLoading')}
      value="menu-part-placeholder"
      onSelect={ignoreSelect}
    />
  );
};

const ICON_ITEM_TOOLTIP_CONTENT_PROPS = { fontSize: 'xs' } as const;
const ICON_ITEM_TOOLTIP_POSITIONING_PROPS = { placement: 'top' } as const;

export interface MenuIconItemProps {
  value: string;
  label: string;
  icon: LucideIcon;
  /** Lucide icons are stroke-only, so `'currentColor'` is how an on state reads. */
  iconFill?: string;
  tone?: 'danger';
  disabled?: boolean;
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
  icon: LucideIcon;
}

export interface IconRadioMenuProps<Value extends string> {
  label: string;
  /** The trigger's accessible name; name the current choice, since the trigger shows only its icon. */
  triggerLabel: string;
  tooltip: ReactNode;
  options: readonly IconRadioMenuOption<Value>[];
  value: Value;
  isActive?: boolean;
  size?: 'sm' | 'md';
  triggerId?: string;
  triggerRef?: Ref<HTMLButtonElement>;
  triggerAttributes?: Readonly<Record<`data-${string}`, string>>;
  onValueChange: (value: Value) => void;
}

const ICON_RADIO_MENU_POSITIONING = { placement: 'bottom-end' } as const;

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

/** The layer stack would dismiss a successor registered before its predecessor's exit animation ends as nested. */
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
