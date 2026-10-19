import type { KeyboardEvent as ReactKeyboardEvent, ReactNode } from 'react';

import { Box, chakra, HStack, Text } from '@chakra-ui/react';
import { Fragment, useCallback } from 'react';

import { Tooltip } from './Tooltip';

const TAB_HOVER_PROPS = { bg: 'bg.hover', color: 'fg' };
const TAB_SHOWN_BG = 'gray.hoverTint/15';
// A hovered tab's background replaces its neighbouring dividers, as the shown tab's does.
const TABLIST_CSS = {
  '& > [role="tab"]:hover + [data-segment-divider], & > [data-segment-divider]:has(+ [role="tab"]:hover)': {
    opacity: 0,
  },
};

/** The strip's fixed height; collapsed blocks and drag snaps size against it. */
export const SEGMENT_TABS_HEIGHT_PX = 40;

export interface SegmentTab<T extends string = string> {
  id: T;
  /** Usually a string; gallery tabs carry a dimmed count span. */
  label: ReactNode;
  /** The tab's name when its label is an icon; it is also shown as the tab's tooltip. */
  ariaLabel?: string;
}

/** Roving focus for a horizontal tablist: arrows cycle, Home/End jump. */
const focusSibling = (event: ReactKeyboardEvent<HTMLElement>) => {
  if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight' && event.key !== 'Home' && event.key !== 'End') {
    return;
  }
  const tabs = [...event.currentTarget.querySelectorAll<HTMLElement>('[role="tab"]')];
  const current = tabs.indexOf(document.activeElement as HTMLElement);
  if (current === -1) {
    return;
  }
  event.preventDefault();
  const next =
    event.key === 'Home'
      ? 0
      : event.key === 'End'
        ? tabs.length - 1
        : (current + (event.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length;
  tabs[next]?.focus();
};

/**
 * Wire caller-owned panels with segmentTabsPanelId/segmentTabsTabId. Active-tab clicks call onSelect again;
 * showActivePanel=false retains selection. trailing sits outside the tablist.
 */
export const SegmentTabs = <T extends string>({
  activeId,
  ariaLabel,
  idBase,
  isCompact = false,
  onSelect,
  showActivePanel = true,
  tabs,
  trailing,
}: {
  activeId: T;
  ariaLabel: string;
  idBase: string;
  /** Embedded strips (popovers, dialog headers) drop the panel-strip height and outer padding. */
  isCompact?: boolean;
  onSelect: (id: T) => void;
  showActivePanel?: boolean;
  tabs: readonly SegmentTab<T>[];
  trailing?: ReactNode;
}) => (
  <HStack
    align="center"
    flexShrink={0}
    gap="0.5"
    h={isCompact ? '8' : `${SEGMENT_TABS_HEIGHT_PX}px`}
    minW="0"
    px={isCompact ? '0' : '1.5'}
  >
    <HStack
      aria-label={ariaLabel}
      aria-orientation="horizontal"
      css={TABLIST_CSS}
      flex="1"
      gap="0.5"
      minW="0"
      overflow="hidden"
      role="tablist"
      onKeyDown={focusSibling}
    >
      {tabs.map((tab, index) => (
        <Fragment key={tab.id}>
          {index > 0 ? (
            <Box
              aria-hidden
              bg="border.emphasized"
              data-segment-divider=""
              flexShrink={0}
              h="3.5"
              // Only the shown tab's background replaces its neighbouring dividers; a collapsed strip keeps them all.
              opacity={showActivePanel && (tab.id === activeId || tabs[index - 1]!.id === activeId) ? 0 : 1}
              rounded="full"
              transition="opacity var(--wb-motion-duration-fast)"
              w="1px"
            />
          ) : null}
          <SegmentTabButton
            ariaLabel={tab.ariaLabel}
            id={tab.id}
            idBase={idBase}
            isSelected={tab.id === activeId}
            isShown={tab.id === activeId && showActivePanel}
            label={tab.label}
            onSelect={onSelect}
          />
        </Fragment>
      ))}
    </HStack>
    {trailing}
  </HStack>
);

export const segmentTabsPanelId = (idBase: string): string => `${idBase}-panel`;
export const segmentTabsTabId = (idBase: string, tabId: string): string => `${idBase}-tab-${tabId}`;

const SegmentTabButton = <T extends string>({
  ariaLabel,
  id,
  idBase,
  isSelected,
  isShown,
  label,
  onSelect,
}: {
  ariaLabel?: string;
  id: T;
  idBase: string;
  isSelected: boolean;
  /** Selected AND its panel is visible; a collapsed block keeps selection without the shown look. */
  isShown: boolean;
  label: ReactNode;
  onSelect: (id: T) => void;
}) => {
  const select = useCallback(() => onSelect(id), [id, onSelect]);

  const button = (
    <chakra.button
      aria-controls={isShown ? segmentTabsPanelId(idBase) : undefined}
      aria-label={ariaLabel}
      aria-selected={isSelected}
      bg={isShown ? TAB_SHOWN_BG : 'transparent'}
      color={isShown ? 'fg' : 'fg.muted'}
      fontSize="md"
      fontWeight="medium"
      h="7"
      id={segmentTabsTabId(idBase, id)}
      minW="10"
      overflow="hidden"
      px="2.5"
      role="tab"
      rounded="control"
      tabIndex={isSelected ? 0 : -1}
      transition="background var(--wb-motion-duration-fast), color var(--wb-motion-duration-fast)"
      type="button"
      _hover={isShown ? undefined : TAB_HOVER_PROPS}
      onClick={select}
    >
      <Text as="span" truncate>
        {label}
      </Text>
    </chakra.button>
  );

  return ariaLabel ? <Tooltip content={ariaLabel}>{button}</Tooltip> : button;
};
