import type { ChakraProps, ThemeTypings } from '@chakra-ui/react';
import { InvButton } from 'common/components/InvButton/InvButton';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import type { ReactElement, ReactNode } from 'react';
import { memo } from 'react';

type Props = {
  label: string;
  tooltip: ReactNode;
  icon?: ReactElement;
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
      <InvIconButton
        aria-label={label}
        tooltip={tooltip}
        icon={icon}
        onClick={onClick}
        isDisabled={isDisabled}
        colorScheme={colorScheme}
        isLoading={isLoading}
        sx={sx}
        data-testid={label}
      />
    );
  }

  return (
    <InvButton
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
      data-testid={label}
    >
      {label}
    </InvButton>
  );
};

export default memo(QueueButton);
