import type { MenuItemProps } from '@invoke-ai/ui-library';
import { Flex, MenuItem } from '@invoke-ai/ui-library';
import { IAITooltip } from 'common/components/IAITooltip';
import type { ReactNode } from 'react';

type Props = MenuItemProps & {
  tooltip?: ReactNode;
  icon: ReactNode;
};

export const IconMenuItem = ({ tooltip, icon, ...props }: Props) => {
  return (
    <IAITooltip label={tooltip} placement="top" gutter={12}>
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
    </IAITooltip>
  );
};

export const IconMenuItemGroup = ({ children }: { children: ReactNode }) => {
  return (
    <Flex gap={2} justifyContent="space-between">
      {children}
    </Flex>
  );
};
