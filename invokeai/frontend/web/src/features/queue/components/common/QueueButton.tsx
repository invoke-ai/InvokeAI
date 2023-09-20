import { ChakraProps, ThemeTypings } from '@chakra-ui/react';
import IAIButton from 'common/components/IAIButton';
import IAIIconButton from 'common/components/IAIIconButton';
import { ReactElement, ReactNode, memo } from 'react';

type Props = {
  label: string;
  tooltip: ReactNode;
  icon: ReactElement;
  onClick?: () => void;
  isDisabled?: boolean;
  colorScheme: ThemeTypings['colorSchemes'];
  asIconButton?: boolean;
  isLoading?: boolean;
  loadingText?: string;
  sx?: ChakraProps['sx'];
};

const QueueButton = ({
  label,
  tooltip,
  icon,
  onClick,
  isDisabled,
  colorScheme,
  asIconButton,
  isLoading,
  loadingText,
  sx,
}: Props) => {
  if (asIconButton) {
    return (
      <IAIIconButton
        aria-label={label}
        tooltip={tooltip}
        icon={icon}
        onClick={onClick}
        isDisabled={isDisabled}
        colorScheme={colorScheme}
        isLoading={isLoading}
        sx={sx}
      />
    );
  }

  return (
    <IAIButton
      aria-label={label}
      tooltip={tooltip}
      leftIcon={icon}
      onClick={onClick}
      isDisabled={isDisabled}
      colorScheme={colorScheme}
      isLoading={isLoading}
      loadingText={loadingText ?? label}
      flexGrow={1}
      sx={sx}
    >
      {label}
    </IAIButton>
  );
};

export default memo(QueueButton);
