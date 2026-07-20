import type { WidgetIconComponent } from '@workbench/widgetContracts';

import { Icon, type IconProps } from '@chakra-ui/react';
import { SquareIcon } from 'lucide-react';

export const WidgetIcon = ({ icon, ...props }: IconProps & { icon?: WidgetIconComponent }) => {
  const ResolvedIcon = icon ?? SquareIcon;

  return <Icon as={ResolvedIcon} {...props} />;
};
