import { defineStyle, defineStyleConfig } from '@chakra-ui/react';

const baseStyle = defineStyle((props) => ({
  fontSize: 9,
  px: 2,
  py: 1,
  minW: 4,
  lineHeight: 1,
  borderRadius: 'sm',
  bg: `${props.colorScheme}.200`,
  color: 'base.900',
  fontWeight: 'bold',
  letterSpacing: 0.6,
  wordBreak: 'break-all',
  whiteSpace: 'nowrap',
  textOverflow: 'ellipsis',
  overflow: 'hidden',
}));

export const badgeTheme = defineStyleConfig({
  baseStyle,
  defaultProps: {
    colorScheme: 'base',
  },
});
