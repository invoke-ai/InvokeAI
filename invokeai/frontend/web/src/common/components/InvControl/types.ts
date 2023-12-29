import type {
  FormControlProps as ChakraFormControlProps,
  FormLabelProps as ChakraFormLabelProps,
} from '@chakra-ui/react';
import type { Feature } from 'common/components/IAIInformationalPopover/constants';

export type InvControlProps = ChakraFormControlProps & {
  label?: string;
  helperText?: string;
  error?: string;
  feature?: Feature;
  renderInfoPopoverInPortal?: boolean;
  labelProps?: Omit<
    InvLabelProps,
    'children' | 'feature' | 'renderInfoPopoverInPortal'
  >;
};

export type InvLabelProps = ChakraFormLabelProps & {
  feature?: Feature;
  renderInfoPopoverInPortal?: boolean;
};
