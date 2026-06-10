import { Tabs as ChakraTabs } from '@chakra-ui/react';
import type { ComponentProps } from 'react';

type TabsRootProps = ComponentProps<typeof ChakraTabs.Root>;

const Root = (props: TabsRootProps) => <ChakraTabs.Root colorPalette="theme" {...props} />;

export const Tabs = {
  ...ChakraTabs,
  Root,
};
