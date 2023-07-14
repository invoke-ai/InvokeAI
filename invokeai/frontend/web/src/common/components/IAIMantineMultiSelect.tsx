import { Tooltip } from '@chakra-ui/react';
import { MultiSelect, MultiSelectProps } from '@mantine/core';
import { useAppDispatch } from 'app/store/storeHooks';
import { shiftKeyPressed } from 'features/ui/store/hotkeysSlice';
import { useMantineMultiSelectStyles } from 'mantine-theme/hooks/useMantineMultiSelectStyles';
import { KeyboardEvent, RefObject, memo, useCallback } from 'react';

type IAIMultiSelectProps = MultiSelectProps & {
  tooltip?: string;
  inputRef?: RefObject<HTMLInputElement>;
};

const IAIMantineMultiSelect = (props: IAIMultiSelectProps) => {
  const { searchable = true, tooltip, inputRef, ...rest } = props;
  const dispatch = useAppDispatch();

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

  const styles = useMantineMultiSelectStyles();

  return (
    <Tooltip label={tooltip} placement="top" hasArrow isOpen={true}>
      <MultiSelect
        ref={inputRef}
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

export default memo(IAIMantineMultiSelect);
