import { FormControl, FormLabel, Tooltip } from '@chakra-ui/react';
import { Select, SelectProps } from '@mantine/core';
import { useAppDispatch } from 'app/store/storeHooks';
import { shiftKeyPressed } from 'features/ui/store/hotkeysSlice';
import { useMantineSelectStyles } from 'mantine-theme/hooks/useMantineSelectStyles';
import { KeyboardEvent, RefObject, memo, useCallback, useState } from 'react';

export type IAISelectDataType = {
  value: string;
  label: string;
  tooltip?: string;
};

type IAISelectProps = Omit<SelectProps, 'label'> & {
  tooltip?: string;
  label?: string;
  inputRef?: RefObject<HTMLInputElement>;
};

const IAIMantineSearchableSelect = (props: IAISelectProps) => {
  const {
    searchable = true,
    tooltip,
    inputRef,
    onChange,
    label,
    disabled,
    ...rest
  } = props;
  const dispatch = useAppDispatch();

  const [searchValue, setSearchValue] = useState('');

  // we want to capture shift keypressed even when an input is focused
  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.shiftKey) {
        dispatch(shiftKeyPressed(true));
      }
    },
    [dispatch]
  );

  const handleKeyUp = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (!e.shiftKey) {
        dispatch(shiftKeyPressed(false));
      }
    },
    [dispatch]
  );

  // wrap onChange to clear search value on select
  const handleChange = useCallback(
    (v: string | null) => {
      // cannot figure out why we were doing this, but it was causing an issue where if you
      // select the currently-selected item, it reset the search value to empty
      // setSearchValue('');

      if (!onChange) {
        return;
      }

      onChange(v);
    },
    [onChange]
  );

  const styles = useMantineSelectStyles();

  return (
    <Tooltip label={tooltip} placement="top" hasArrow>
      <Select
        ref={inputRef}
        label={
          label ? (
            <FormControl isDisabled={disabled}>
              <FormLabel>{label}</FormLabel>
            </FormControl>
          ) : undefined
        }
        disabled={disabled}
        searchValue={searchValue}
        onSearchChange={setSearchValue}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        onKeyUp={handleKeyUp}
        searchable={searchable}
        maxDropdownHeight={300}
        styles={styles}
        {...rest}
      />
    </Tooltip>
  );
};

export default memo(IAIMantineSearchableSelect);
