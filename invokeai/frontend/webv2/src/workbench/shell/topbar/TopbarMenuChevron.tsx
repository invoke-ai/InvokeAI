import { Icon } from '@chakra-ui/react';
import { ChevronDownIcon } from 'lucide-react';

const BRIGHTENED_PROPS = { color: 'fg' } as const;

interface TopbarMenuChevronProps {
  /** Take the trigger button's own icon size, for a chevron that is the trigger's only glyph. */
  fillTrigger?: boolean;
}

/** The trigger's dropdown chevron; it brightens with its trigger, which must carry `className="group"`. */
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
