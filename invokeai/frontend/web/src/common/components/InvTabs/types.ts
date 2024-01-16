import type {
  TabListProps as InvTabListProps,
  TabPanelProps as InvTabPanelProps,
  TabPanelsProps as InvTabPanelsProps,
  TabProps as ChakraTabProps,
  TabsProps as InvTabsProps,
} from '@chakra-ui/react';

export type {
  InvTabListProps,
  InvTabPanelProps,
  InvTabPanelsProps,
  InvTabsProps,
};

export type InvTabProps = ChakraTabProps & {
  badges?: (string | number)[];
};
