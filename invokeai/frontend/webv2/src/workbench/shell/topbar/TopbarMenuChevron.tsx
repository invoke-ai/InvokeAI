import { Icon } from '@chakra-ui/react';
import { ChevronDownIcon } from 'lucide-react';

const BRIGHTENED_PROPS = { color: 'fg' } as const;

interface TopbarMenuChevronProps {
  /** Take the trigger button's own icon size, for a chevron that is the trigger's only glyph. */
  fillTrigger?: boolean;
}

/**
 * The dropdown chevron of a topbar menu trigger: a 12px glyph beside the trigger's content unless `fillTrigger`. It
 * brightens while its trigger is hovered or its menu is open, so the trigger must carry `className="group"`.
 */
export const TopbarMenuChevron = ({ fillTrigger = false }: TopbarMenuChevronProps) => (
  <Icon
    as={ChevronDownIcon}
    boxSize={fillTrigger ? undefined : '3'}
    color="fg.subtle"
    flexShrink={0}
    transition="color var(--wb-motion-duration-fast) ease"
    _groupHover={BRIGHTENED_PROPS}
    _groupExpanded={BRIGHTENED_PROPS}
  />
);
