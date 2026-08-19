import type { ElementType } from 'react';

import { Icon, Menu, Stack, Text } from '@chakra-ui/react';

/**
 * The one menu-item composition every workflow dropdown renders through: a
 * leading glyph, a label, and — for menus whose choices need explaining — a
 * second hint line under it. The graph preview's "Open as" menu and the
 * library rail's overflow menu keep their own, distinct item *sets*; what they
 * share is this layout, which was hand-rolled twice and drifted (icon
 * alignment, padding, disabled opacity) between the two.
 *
 * A hint switches the row to two lines, so the icon top-aligns with the label
 * instead of floating centered against both.
 */

const DISABLED_ITEM = { opacity: 0.4 } as const;
const SINGLE_LINE_LAYOUT = { alignItems: 'center', gap: '2.5', py: '1' } as const;
const TWO_LINE_LAYOUT = { alignItems: 'flex-start', gap: '2.5', py: '1.5' } as const;
const SINGLE_LINE_ICON = { boxSize: '3.5', flexShrink: 0 } as const;
const TWO_LINE_ICON = { boxSize: '3.5', flexShrink: 0, mt: '0.5' } as const;

export interface MenuActionItemProps {
  /** Second line under the label, for choices whose consequences are not obvious. */
  hint?: string;
  icon: ElementType;
  isDisabled?: boolean;
  label: string;
  /** `'danger'` is reserved for destructive choices (delete). */
  tone?: 'danger';
  /** Ark's item value, also mirrored to `data-menu-item` so tests can address the row. */
  value: string;
  onSelect: () => void;
}

export const MenuActionItem = ({ hint, icon, isDisabled, label, tone, value, onSelect }: MenuActionItemProps) => (
  <Menu.Item
    {...(hint ? TWO_LINE_LAYOUT : SINGLE_LINE_LAYOUT)}
    _disabled={DISABLED_ITEM}
    color={tone === 'danger' ? 'fg.error' : undefined}
    data-menu-item={value}
    disabled={isDisabled}
    value={value}
    onClick={onSelect}
  >
    <Icon as={icon} {...(hint ? TWO_LINE_ICON : SINGLE_LINE_ICON)} />
    {hint ? (
      <Stack gap="0" minW="0">
        <Menu.ItemText>{label}</Menu.ItemText>
        <Text color="fg.subtle" fontSize="2xs">
          {hint}
        </Text>
      </Stack>
    ) : (
      <Menu.ItemText>{label}</Menu.ItemText>
    )}
  </Menu.Item>
);
