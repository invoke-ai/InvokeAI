import type { ComponentProps } from 'react';

import { Tabs as ChakraTabs } from '@chakra-ui/react';

type TabsRootProps = ComponentProps<typeof ChakraTabs.Root>;

const Root = (props: TabsRootProps) => <ChakraTabs.Root colorPalette="accent" {...props} />;

export const Tabs = {
  ...ChakraTabs,
  Root,
};
