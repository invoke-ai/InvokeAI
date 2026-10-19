import type { TextareaProps } from '@invoke-ai/ui-library';
import { chakra, forwardRef, typedMemo, useStyleConfig } from '@invoke-ai/ui-library';
import type { ComponentProps } from 'react';
import TextareaAutosize from 'react-textarea-autosize';

const ChakraTextareaAutosize = chakra(TextareaAutosize);

export const AutosizeTextarea = typedMemo(
  forwardRef<ComponentProps<typeof ChakraTextareaAutosize> & TextareaProps, typeof ChakraTextareaAutosize>(
    ({ variant, ...rest }, ref) => {
      const styles = useStyleConfig('Textarea', { variant });
      return <ChakraTextareaAutosize __css={styles} ref={ref} {...rest} />;
    }
  )
);
AutosizeTextarea.displayName = 'AutosizeTextarea';
