import type { WidgetIconComponent } from '@workbench/widgetContracts';

import { Box, Spinner } from '@chakra-ui/react';
import { WidgetIcon } from '@workbench/iconResolver';

export const WidgetIdentityIcon = ({ icon, isLoading = false }: { icon: WidgetIconComponent; isLoading?: boolean }) => (
  <Box aria-hidden="true" boxSize="3" data-widget-identity-slot="" flexShrink="0">
    {isLoading ? (
      <Spinner borderWidth="1.5px" boxSize="full" color="fg.subtle" display="block" />
    ) : (
      <WidgetIcon boxSize="full" display="block" icon={icon} />
    )}
  </Box>
);
