import { Icon } from '@chakra-ui/react';
import { IconButton, Tooltip } from '@platform/ui';
import { SparklesIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

export const ModelDefaultButton = ({
  active,
  disabled,
  label,
  onClick,
}: {
  active?: boolean;
  disabled?: boolean;
  label?: string;
  onClick: () => void;
}) => {
  const { t } = useTranslation();
  const resolvedLabel = label ?? t('widgets.generate.useModelDefault');
  const isActive = active ?? !disabled;
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
      <IconButton
        aria-label={resolvedLabel}
        colorPalette={isActive ? 'brand' : undefined}
        disabled={disabled}
        size="2xs"
        variant="ghost"
        onClick={handleClick}
      >
        <Icon as={SparklesIcon} boxSize="2.5" />
      </IconButton>
    </Tooltip>
  );
};
