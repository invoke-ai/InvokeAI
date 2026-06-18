import { Icon, type IconProps } from '@chakra-ui/react';
import { SquareIcon } from 'lucide-react';

import type { WidgetIconComponent } from './types';

export const WidgetIcon = ({ icon, ...props }: IconProps & { icon?: WidgetIconComponent }) => {
  const ResolvedIcon = icon ?? SquareIcon;

  return <Icon as={ResolvedIcon} {...props} />;
};
