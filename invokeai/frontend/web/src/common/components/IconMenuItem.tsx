import type { MenuItemProps } from '@invoke-ai/ui-library';
import { Flex, MenuItem, Tooltip } from '@invoke-ai/ui-library';
import type { ReactNode } from 'react';

type Props = MenuItemProps & {
  tooltip?: ReactNode;
  icon: ReactNode;
};

export const IconMenuItem = ({ tooltip, icon, ...props }: Props) => {
  return (
    <Tooltip label={tooltip} placement="top" gutter={12}>
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
  return <Flex gap={2}>{children}</Flex>;
};
