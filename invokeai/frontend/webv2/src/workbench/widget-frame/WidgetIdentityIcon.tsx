import type { WidgetIconComponent } from '@workbench/widgetContracts';

import { Box, Spinner } from '@chakra-ui/react';
import { WidgetIcon } from '@workbench/iconResolver';

export const WidgetIdentityIcon = ({
  boxSize = '3',
  icon,
  isLoading = false,
}: {
  /** The slot's size; the icon and the spinner both fill it. */
  boxSize?: string;
  icon: WidgetIconComponent;
  isLoading?: boolean;
}) => (
  <Box aria-hidden="true" boxSize={boxSize} data-widget-identity-slot="" flexShrink="0">
    {isLoading ? (
      <Spinner borderWidth="1.5px" boxSize="full" color="fg.subtle" display="block" />
    ) : (
      <WidgetIcon boxSize="full" display="block" icon={icon} />
    )}
  </Box>
);
