import type { ChakraProps, ThemeTypings } from '@invoke-ai/ui-library';
import { Button, IconButton } from '@invoke-ai/ui-library';
import type { ReactElement, ReactNode } from 'react';
import { memo } from 'react';

type Props = {
  label: string;
  tooltip: ReactNode;
  icon?: ReactElement;
  onPointerUp?: () => void;
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
  onPointerUp,
  isDisabled,
  colorScheme,
  asIconButton,
  isLoading,
  loadingText,
  sx,
}: Props) => {
  if (asIconButton) {
    return (
      <IconButton
        aria-label={label}
        tooltip={tooltip}
        icon={icon}
        onPointerUp={onPointerUp}
        isDisabled={isDisabled}
        colorScheme={colorScheme}
        isLoading={isLoading}
        sx={sx}
        data-testid={label}
      />
    );
  }

  return (
    <Button
      aria-label={label}
      tooltip={tooltip}
      leftIcon={icon}
      onPointerUp={onPointerUp}
      isDisabled={isDisabled}
      colorScheme={colorScheme}
      isLoading={isLoading}
      loadingText={loadingText ?? label}
      flexGrow={1}
      sx={sx}
      data-testid={label}
    >
      {label}
    </Button>
  );
};

export default memo(QueueButton);
