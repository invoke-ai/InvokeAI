import { useColorMode } from '@chakra-ui/react';
import { TextInput, TextInputProps } from '@mantine/core';
import { useChakraThemeTokens } from 'common/hooks/useChakraThemeTokens';
import { useCallback } from 'react';
import { mode } from 'theme/util/mode';

type IAIMantineTextInputProps = TextInputProps;

export default function IAIMantineTextInput(props: IAIMantineTextInputProps) {
  const { ...rest } = props;
  const {
    base50,
    base100,
    base200,
    base300,
    base800,
    base700,
    base900,
    accent500,
    accent300,
  } = useChakraThemeTokens();
  const { colorMode } = useColorMode();

  const stylesFunc = useCallback(
    () => ({
      input: {
        color: mode(base900, base100)(colorMode),
        backgroundColor: mode(base50, base900)(colorMode),
        borderColor: mode(base200, base800)(colorMode),
        borderWidth: 2,
        outline: 'none',
        ':focus': {
          borderColor: mode(accent300, accent500)(colorMode),
        },
      },
      label: {
        color: mode(base700, base300)(colorMode),
        fontWeight: 'normal' as const,
        marginBottom: 4,
      },
    }),
    [
      accent300,
      accent500,
      base100,
      base200,
      base300,
      base50,
      base700,
      base800,
      base900,
      colorMode,
    ]
  );

  return <TextInput styles={stylesFunc} {...rest} />;
}
