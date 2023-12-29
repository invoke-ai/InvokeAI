import type { ChakraProps } from '@chakra-ui/react';
import type {
  ChakraStylesConfig,
  GroupBase,
  OptionBase,
  Props as ChakraReactSelectProps,
  SelectInstance,
  SingleValue,
} from 'chakra-react-select';
import type { ReactNode } from 'react';
export type {} from 'react-select/base';

export interface InvSelectOption extends OptionBase {
  label: string;
  value: string;
  description?: string;
  icon?: ReactNode;
  tooltip?: string;
  tags?: string[];
}

export type InvSelectProps = ChakraReactSelectProps<
  InvSelectOption,
  false,
  GroupBase<InvSelectOption>
> & {
  sx?: ChakraProps['sx'];
  selectRef?: React.RefObject<
    SelectInstance<InvSelectOption, false, GroupBase<InvSelectOption>>
  >;
};
export type CustomChakraStylesConfig = ChakraStylesConfig<
  InvSelectOption,
  false,
  GroupBase<InvSelectOption>
>;

export type InvSelectFallbackProps = {
  label: string;
};

export type InvSelectOnChange = (
  v: SingleValue<InvSelectOption> | null
) => void;
