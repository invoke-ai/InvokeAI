import { Icon } from '@chakra-ui/react';
import { IconButton, Tooltip } from '@platform/ui';
import { RotateCcwIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * `InputGroup endElementProps` for fields hosting this button: interactive,
 * and padded for an icon button rather than Chakra's text-glyph default
 * (`px: 3`), which floats the button too far off the field's end edge.
 */
export const MODEL_DEFAULT_END_ELEMENT_PROPS = { pointerEvents: 'auto', px: '1' } as const;

/**
 * Reset-to-model-default affordance. Callers render it only while the value
 * differs from the model default, so its presence itself signals "modified".
 */
export const ModelDefaultButton = ({ label, onClick }: { label?: string; onClick: () => void }) => {
  const { t } = useTranslation();
  const resolvedLabel = label ?? t('widgets.generate.useModelDefault');
  const handleClick = useCallback(
    (event: React.MouseEvent<HTMLButtonElement>) => {
      event.preventDefault();
      event.stopPropagation();
      onClick();
    },
    [onClick]
  );

  return (
    <Tooltip content={resolvedLabel}>
      <IconButton aria-label={resolvedLabel} color="fg.muted" size="2xs" variant="ghost" onClick={handleClick}>
        <Icon as={RotateCcwIcon} boxSize="2.5" />
      </IconButton>
    </Tooltip>
  );
};
