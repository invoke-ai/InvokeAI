import type {
  GroupBase,
  SelectComponentsConfig,
  StylesConfig,
} from 'chakra-react-select';
import { Select as ChakraReactSelect } from 'chakra-react-select';
import { memo, useMemo } from 'react';

import { CustomMenuList } from './CustomMenuList';
import { CustomOption } from './CustomOption';
import type {
  CustomChakraStylesConfig,
  InvSelectOption,
  InvSelectProps,
} from './types';

const styles: StylesConfig<InvSelectOption> = {
  menuPortal: (provided) => ({ ...provided, zIndex: 9999 }),
};

const components: SelectComponentsConfig<
  InvSelectOption,
  false,
  GroupBase<InvSelectOption>
> = {
  Option: CustomOption,
  MenuList: CustomMenuList,
  // Menu: CustomMenu,
};

export const InvSelect = memo((props: InvSelectProps) => {
  const { sx, selectRef, ...rest } = props;
  const chakraStyles = useMemo<CustomChakraStylesConfig>(
    () => ({
      container: (provided, _state) => ({ ...provided, w: 'full', ...sx }),
      option: (provided, _state) => ({ ...provided, p: 0 }),
      indicatorsContainer: (provided, _state) => ({
        ...provided,
        w: 8,
        alignItems: 'center',
        justifyContent: 'center',
        '> div': { p: 0, w: 'full', h: 'full', bg: 'unset' },
      }),
      dropdownIndicator: (provided, _state) => ({
        ...provided,
        display:
          _state.hasValue && _state.selectProps.isClearable ? 'none' : 'flex',
      }),
      crossIcon: (provided, _state) => ({ ...provided, boxSize: 2.5 }),
      inputContainer: (provided, _state) => ({
        ...provided,
        cursor: 'pointer',
      }),
    }),
    [sx]
  );

  return (
    <ChakraReactSelect<InvSelectOption, false, GroupBase<InvSelectOption>>
      ref={selectRef}
      menuPortalTarget={document.body}
      colorScheme="base"
      selectedOptionColorScheme="base"
      components={components}
      chakraStyles={chakraStyles}
      styles={styles}
      variant="filled"
      menuPosition="fixed"
      {...rest}
    />
  );
});

InvSelect.displayName = 'InvSelect';
