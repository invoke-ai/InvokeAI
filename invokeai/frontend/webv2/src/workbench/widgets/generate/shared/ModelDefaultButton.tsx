import { Icon } from '@chakra-ui/react';
import { IconButton, Tooltip } from '@workbench/components/ui';
import { SparklesIcon } from 'lucide-react';
import { useCallback } from 'react';

export const ModelDefaultButton = ({
  active,
  disabled,
  label = 'Use model default',
  onClick,
}: {
  active?: boolean;
  disabled?: boolean;
  label?: string;
  onClick: () => void;
}) => {
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
    <Tooltip content={label}>
      <IconButton
        aria-label={label}
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
