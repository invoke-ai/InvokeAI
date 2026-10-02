import type { ComponentProps } from 'react';

import { Tabs as ChakraTabs } from '@chakra-ui/react';

type ChakraTabsRootProps = ComponentProps<typeof ChakraTabs.Root>;
type TabsRootProps = Omit<ChakraTabsRootProps, 'size'> & {
  size?: ChakraTabsRootProps['size'] | 'xs';
};

const Root = ({ size, ...props }: TabsRootProps) => (
  <ChakraTabs.Root colorPalette="accent" size={size as ChakraTabsRootProps['size']} {...props} />
);

export const Tabs = {
  ...ChakraTabs,
  Root,
};
