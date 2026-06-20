import { Icon } from '@chakra-ui/react';
import { IconButton, Tooltip } from '@workbench/components/ui';
import { SparklesIcon } from 'lucide-react';

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

  return (
    <Tooltip content={label}>
      <IconButton
        aria-label={label}
        colorPalette={isActive ? 'brand' : undefined}
        disabled={disabled}
        size="2xs"
        variant="ghost"
        onClick={(event) => {
          event.preventDefault();
          event.stopPropagation();
          onClick();
        }}
      >
        <Icon as={SparklesIcon} boxSize="2.5" />
      </IconButton>
    </Tooltip>
  );
};
