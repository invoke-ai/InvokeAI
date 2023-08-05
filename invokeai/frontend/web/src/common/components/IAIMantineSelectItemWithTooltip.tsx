import { Box, Tooltip } from '@chakra-ui/react';
import { Text } from '@mantine/core';
import { forwardRef, memo } from 'react';

interface ItemProps extends React.ComponentPropsWithoutRef<'div'> {
  label: string;
  description?: string;
  tooltip?: string;
  disabled?: boolean;
}

const IAIMantineSelectItemWithTooltip = forwardRef<HTMLDivElement, ItemProps>(
  (
    { label, tooltip, description, disabled: _disabled, ...others }: ItemProps,
    ref
  ) => (
    <Tooltip label={tooltip} placement="top" hasArrow openDelay={500}>
      <Box ref={ref} {...others}>
        <Box>
          <Text>{label}</Text>
          {description && (
            <Text size="xs" color="base.600">
              {description}
            </Text>
          )}
        </Box>
      </Box>
    </Tooltip>
  )
);

IAIMantineSelectItemWithTooltip.displayName = 'IAIMantineSelectItemWithTooltip';

export default memo(IAIMantineSelectItemWithTooltip);
