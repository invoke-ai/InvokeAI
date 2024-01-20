import { Box, forwardRef, Textarea as ChakraTextarea } from '@invoke-ai/ui';
import { useGlobalModifiersSetters } from 'common/hooks/useGlobalModifiers';
import { stopPastePropagation } from 'common/util/stopPastePropagation';
import type { KeyboardEvent } from 'react';
import { memo, useCallback } from 'react';
import ResizeTextarea from 'react-textarea-autosize';

import type { InvAutosizeTextareaProps } from './types';

export const InvAutosizeTextarea = memo(
  forwardRef<InvAutosizeTextareaProps, typeof ResizeTextarea>(
    (props: InvAutosizeTextareaProps, ref) => {
      const { setShift } = useGlobalModifiersSetters();
      const onKeyUpDown = useCallback(
        (e: KeyboardEvent<HTMLTextAreaElement>) => {
          setShift(e.shiftKey);
        },
        [setShift]
      );
      return (
        <Box pos="relative">
          <ChakraTextarea
            as={ResizeTextarea}
            ref={ref}
            overflow="scroll"
            w="100%"
            minRows={3}
            minH={20}
            onPaste={stopPastePropagation}
            onKeyUp={onKeyUpDown}
            onKeyDown={onKeyUpDown}
            {...props}
          />
        </Box>
      );
    }
  )
);

InvAutosizeTextarea.displayName = 'InvAutosizeTextarea';
