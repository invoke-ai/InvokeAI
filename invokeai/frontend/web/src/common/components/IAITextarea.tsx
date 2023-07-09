import { Textarea, TextareaProps, forwardRef } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { stopPastePropagation } from 'common/util/stopPastePropagation';
import { shiftKeyPressed } from 'features/ui/store/hotkeysSlice';
import { KeyboardEvent, memo, useCallback } from 'react';

const IAITextarea = forwardRef((props: TextareaProps, ref) => {
  const dispatch = useAppDispatch();
  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.shiftKey) {
        dispatch(shiftKeyPressed(true));
      }
    },
    [dispatch]
  );

  const handleKeyUp = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (!e.shiftKey) {
        dispatch(shiftKeyPressed(false));
      }
    },
    [dispatch]
  );

  return (
    <Textarea
      ref={ref}
      onPaste={stopPastePropagation}
      onKeyDown={handleKeyDown}
      onKeyUp={handleKeyUp}
      {...props}
    />
  );
});

export default memo(IAITextarea);
