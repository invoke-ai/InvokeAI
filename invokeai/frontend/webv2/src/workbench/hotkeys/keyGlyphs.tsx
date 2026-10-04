import type { ReactNode } from 'react';

import { chakra, Icon } from '@chakra-ui/react';
import {
  ArrowBigUpIcon,
  ArrowDownIcon,
  ArrowLeftIcon,
  ArrowRightIcon,
  ArrowRightToLineIcon,
  ArrowUpIcon,
  ChevronUpIcon,
  CommandIcon,
  CornerDownLeftIcon,
  DeleteIcon,
  OptionIcon,
  type LucideIcon,
} from 'lucide-react';

import { IS_MAC_OS } from './keys';

/** Use universal key icons everywhere; macOS modifier glyphs become word labels on Windows/Linux. */
const UNIVERSAL_KEY_ICONS: Record<string, LucideIcon> = {
  arrowdown: ArrowDownIcon,
  arrowleft: ArrowLeftIcon,
  arrowright: ArrowRightIcon,
  arrowup: ArrowUpIcon,
  backspace: DeleteIcon,
  enter: CornerDownLeftIcon,
  tab: ArrowRightToLineIcon,
};

const MAC_MODIFIER_ICONS: Record<string, LucideIcon> = {
  alt: OptionIcon,
  cmd: CommandIcon,
  ctrl: ChevronUpIcon,
  meta: CommandIcon,
  option: OptionIcon,
  shift: ArrowBigUpIcon,
};

/** The icon for a canonical hotkey part (`enter`, `cmd`, …), or null when the part reads better as text. */
export const getShortcutKeyIcon = (part: string, isMacOs: boolean = IS_MAC_OS): LucideIcon | null =>
  UNIVERSAL_KEY_ICONS[part] ?? (isMacOs ? (MAC_MODIFIER_ICONS[part] ?? null) : null);

/** Spoken names for keys whose visual form is an icon. */
export const SHORTCUT_KEY_ARIA_LABELS: Record<string, string> = {
  alt: 'Alt',
  arrowdown: 'Arrow down',
  arrowleft: 'Arrow left',
  arrowright: 'Arrow right',
  arrowup: 'Arrow up',
  backspace: 'Backspace',
  cmd: 'Command',
  ctrl: 'Control',
  enter: 'Enter',
  meta: 'Meta',
  option: 'Alt',
  shift: 'Shift',
  tab: 'Tab',
};

/** Render an icon or the caller's text casing; icon-only hints include a spoken key name. */
export const ShortcutKeyGlyph = ({ fallback, part }: { fallback?: ReactNode; part: string }) => {
  const GlyphIcon = getShortcutKeyIcon(part);

  if (!GlyphIcon) {
    return fallback ?? part;
  }

  return (
    <>
      <Icon aria-hidden="true" as={GlyphIcon} boxSize="2.5" verticalAlign="middle" />
      <chakra.span srOnly>{SHORTCUT_KEY_ARIA_LABELS[part] ?? part}</chakra.span>
    </>
  );
};
