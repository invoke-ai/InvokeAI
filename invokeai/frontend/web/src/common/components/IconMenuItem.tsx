import type { MenuItemProps, TooltipProps } from '@invoke-ai/ui-library';
import { Flex, MenuItem, Tooltip } from '@invoke-ai/ui-library';
import type { ReactNode } from 'react';

type Props = MenuItemProps & {
  tooltip?: ReactNode;
  tooltipPlacement?: TooltipProps['placement'];
  icon: ReactNode;
};

export const IconMenuItem = ({ tooltip, tooltipPlacement = 'top', icon, ...props }: Props) => {
  return (
    <Tooltip label={tooltip} placement={tooltipPlacement} gutter={12}>
      <MenuItem
        display="flex"
        alignItems="center"
        justifyContent="center"
        w="min-content"
        aspectRatio="1"
        borderRadius="base"
        {...props}
      >
        {icon}
      </MenuItem>
    </Tooltip>
  );
};

export const IconMenuItemGroup = ({ children }: { children: ReactNode }) => {
  return (
    <Flex gap={2} justifyContent="space-between">
      {children}
    </Flex>
  );
};
