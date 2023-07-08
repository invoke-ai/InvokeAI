import { Tooltip, useColorMode, useToken } from '@chakra-ui/react';
import { Select, SelectProps } from '@mantine/core';
import { useAppDispatch } from 'app/store/storeHooks';
import { useChakraThemeTokens } from 'common/hooks/useChakraThemeTokens';
import { shiftKeyPressed } from 'features/ui/store/hotkeysSlice';
import { KeyboardEvent, memo, useCallback } from 'react';
import { mode } from 'theme/util/mode';

export type IAISelectDataType = {
  value: string;
  label: string;
  tooltip?: string;
};

type IAISelectProps = SelectProps & {
  tooltip?: string;
};

const IAIMantineSelect = (props: IAISelectProps) => {
  const { searchable = true, tooltip, ...rest } = props;
  const dispatch = useAppDispatch();
  const {
    base50,
    base100,
    base200,
    base300,
    base400,
    base500,
    base600,
    base700,
    base800,
    base900,
    accent200,
    accent300,
    accent400,
    accent500,
    accent600,
  } = useChakraThemeTokens();

  const { colorMode } = useColorMode();

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

  const [boxShadow] = useToken('shadows', ['dark-lg']);

  return (
    <Tooltip label={tooltip} placement="top" hasArrow>
      <Select
        onKeyDown={handleKeyDown}
        onKeyUp={handleKeyUp}
        searchable={searchable}
        styles={() => ({
          label: {
            color: mode(base700, base300)(colorMode),
            fontWeight: 'normal',
          },
          input: {
            backgroundColor: mode(base50, base900)(colorMode),
            borderWidth: '2px',
            borderColor: mode(base200, base800)(colorMode),
            color: mode(base900, base100)(colorMode),
            paddingRight: 24,
            fontWeight: 600,
            '&:hover': { borderColor: mode(base300, base600)(colorMode) },
            '&:focus': {
              borderColor: mode(accent300, accent600)(colorMode),
            },
            '&:is(:focus, :hover)': {
              borderColor: mode(base400, base500)(colorMode),
            },
            '&:focus-within': {
              borderColor: mode(accent200, accent600)(colorMode),
            },
            '&[data-disabled]': {
              backgroundColor: mode(base300, base700)(colorMode),
              color: mode(base600, base400)(colorMode),
              cursor: 'not-allowed',
            },
          },
          value: {
            backgroundColor: mode(base100, base900)(colorMode),
            color: mode(base900, base100)(colorMode),
            button: {
              color: mode(base900, base100)(colorMode),
            },
            '&:hover': {
              backgroundColor: mode(base300, base700)(colorMode),
              cursor: 'pointer',
            },
          },
          dropdown: {
            backgroundColor: mode(base200, base800)(colorMode),
            borderColor: mode(base200, base800)(colorMode),
            boxShadow,
          },
          item: {
            backgroundColor: mode(base200, base800)(colorMode),
            color: mode(base800, base200)(colorMode),
            padding: 6,
            '&[data-hovered]': {
              color: mode(base900, base100)(colorMode),
              backgroundColor: mode(base300, base700)(colorMode),
            },
            '&[data-active]': {
              backgroundColor: mode(base300, base700)(colorMode),
              '&:hover': {
                color: mode(base900, base100)(colorMode),
                backgroundColor: mode(base300, base700)(colorMode),
              },
            },
            '&[data-selected]': {
              backgroundColor: mode(accent400, accent600)(colorMode),
              color: mode(base50, base100)(colorMode),
              fontWeight: 600,
              '&:hover': {
                backgroundColor: mode(accent500, accent500)(colorMode),
                color: mode('white', base50)(colorMode),
              },
            },
            '&[data-disabled]': {
              color: mode(base500, base600)(colorMode),
              cursor: 'not-allowed',
            },
          },
          rightSection: {
            width: 32,
            button: {
              color: mode(base900, base100)(colorMode),
            },
          },
        })}
        {...rest}
      />
    </Tooltip>
  );
};

export default memo(IAIMantineSelect);
