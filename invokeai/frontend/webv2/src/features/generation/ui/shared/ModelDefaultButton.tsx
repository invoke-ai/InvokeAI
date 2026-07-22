import { Icon } from '@chakra-ui/react';
import { IconButton, Tooltip } from '@platform/ui';
import { RotateCcwIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

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
