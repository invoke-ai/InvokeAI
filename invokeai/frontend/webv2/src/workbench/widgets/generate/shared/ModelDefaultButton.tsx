import { Icon } from '@chakra-ui/react';
import { IconButton, Tooltip } from '@workbench/components/ui';
import { SparklesIcon } from 'lucide-react';

export const ModelDefaultButton = ({
  disabled,
  label = 'Use model default',
  onClick,
}: {
  disabled?: boolean;
  label?: string;
  onClick: () => void;
}) => (
  <Tooltip content={label}>
    <IconButton
      aria-label={label}
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
