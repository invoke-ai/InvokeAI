import type { ComponentProps } from 'react';

import { Tabs as ChakraTabs } from '@chakra-ui/react';

const Root = (props: ComponentProps<typeof ChakraTabs.Root>) => <ChakraTabs.Root colorPalette="accent" {...props} />;

export const Tabs = {
  ...ChakraTabs,
  Root,
};
