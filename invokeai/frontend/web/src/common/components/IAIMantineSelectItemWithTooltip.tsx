import { Tooltip, Text } from '@mantine/core';
import { forwardRef, memo } from 'react';

interface ItemProps extends React.ComponentPropsWithoutRef<'div'> {
  label: string;
  description?: string;
  tooltip?: string;
}

const IAIMantineSelectItemWithTooltip = forwardRef<HTMLDivElement, ItemProps>(
  ({ label, tooltip, description, ...others }: ItemProps, ref) => (
    <div ref={ref} {...others}>
      {tooltip ? (
        <Tooltip.Floating label={tooltip}>
          <div>
            <Text>{label}</Text>
            {description && (
              <Text size="xs" color="base.600">
                {description}
              </Text>
            )}
          </div>
        </Tooltip.Floating>
      ) : (
        <div>
          <Text>{label}</Text>
          {description && (
            <Text size="xs" color="base.600">
              {description}
            </Text>
          )}
        </div>
      )}
    </div>
  )
);

IAIMantineSelectItemWithTooltip.displayName = 'IAIMantineSelectItemWithTooltip';

export default memo(IAIMantineSelectItemWithTooltip);
