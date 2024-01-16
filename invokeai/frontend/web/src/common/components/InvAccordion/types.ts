import type { AccordionButtonProps as ChakraAccordionButtonProps } from '@chakra-ui/react';
export type {
  AccordionIconProps as InvAccordionIconProps,
  AccordionItemProps as InvAccordionItemProps,
  AccordionPanelProps as InvAccordionPanelProps,
  AccordionProps as InvAccordionProps,
} from '@chakra-ui/react';

export type InvAccordionButtonProps = ChakraAccordionButtonProps & {
  badges?: (string | number)[];
};
