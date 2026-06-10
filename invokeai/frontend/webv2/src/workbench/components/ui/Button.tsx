import {
  Button as ChakraButton,
  CloseButton as ChakraCloseButton,
  IconButton as ChakraIconButton,
} from '@chakra-ui/react';
import type { ComponentProps } from 'react';

type ButtonProps = ComponentProps<typeof ChakraButton>;
type IconButtonProps = ComponentProps<typeof ChakraIconButton>;
type CloseButtonProps = ComponentProps<typeof ChakraCloseButton>;

export const Button = (props: ButtonProps) => <ChakraButton colorPalette="theme" {...props} />;

export const IconButton = (props: IconButtonProps) => <ChakraIconButton colorPalette="theme" {...props} />;

export const CloseButton = (props: CloseButtonProps) => <ChakraCloseButton colorPalette="theme" {...props} />;
