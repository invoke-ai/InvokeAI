import { useColorMode, useToken } from '@chakra-ui/react';
import { useChakraThemeTokens } from 'common/hooks/useChakraThemeTokens';
import { useCallback } from 'react';
import { mode } from 'theme/util/mode';

export const useMantineMultiSelectStyles = () => {
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
  const [boxShadow] = useToken('shadows', ['dark-lg']);

  const styles = useCallback(
    () => ({
      label: {
        color: mode(base700, base300)(colorMode),
      },
      separatorLabel: {
        color: mode(base500, base500)(colorMode),
        '::after': { borderTopColor: mode(base300, base700)(colorMode) },
      },
      searchInput: {
        ':placeholder': {
          color: mode(base300, base700)(colorMode),
        },
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
        backgroundColor: mode(base200, base800)(colorMode),
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
        width: 24,
        padding: 20,
        button: {
          color: mode(base900, base100)(colorMode),
        },
      },
    }),
    [
      accent200,
      accent300,
      accent400,
      accent500,
      accent600,
      base100,
      base200,
      base300,
      base400,
      base50,
      base500,
      base600,
      base700,
      base800,
      base900,
      boxShadow,
      colorMode,
    ]
  );

  return styles;
};
