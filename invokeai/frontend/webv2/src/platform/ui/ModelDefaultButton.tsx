import { Icon } from '@chakra-ui/react';
import { IconButton } from '@platform/ui/Button';
import { Tooltip } from '@platform/ui/Tooltip';
import { RotateCcwIcon } from 'lucide-react';
import { useCallback } from 'react';

/**
 * `InputGroup endElementProps` for fields hosting this button: interactive,
 * and padded for an icon button rather than Chakra's text-glyph default
 * (`px: 3`), which floats the button too far off the field's end edge.
 */
export const MODEL_DEFAULT_END_ELEMENT_PROPS = { pointerEvents: 'auto', px: '1' } as const;

/**
 * Reset-to-default affordance. Callers render it only while the value
 * differs from the default, so its presence itself signals "modified". No
 * fallback label: every call site knows what it's resetting and says so.
 */
export const ModelDefaultButton = ({ label, onClick }: { label: string; onClick: () => void }) => {
  const handleClick = useCallback(
    (event: React.MouseEvent<HTMLButtonElement>) => {
      event.preventDefault();
      event.stopPropagation();
      onClick();
    },
    [onClick]
  );

  return (
    <Tooltip content={label}>
      <IconButton aria-label={label} color="fg.muted" size="2xs" variant="ghost" onClick={handleClick}>
        <Icon as={RotateCcwIcon} boxSize="2.5" />
      </IconButton>
    </Tooltip>
  );
};
