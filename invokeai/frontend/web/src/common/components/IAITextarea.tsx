import { Textarea, TextareaProps, forwardRef } from '@chakra-ui/react';
import { stopPastePropagation } from 'common/util/stopPastePropagation';
import { memo } from 'react';

const IAITextarea = forwardRef((props: TextareaProps, ref) => {
  return <Textarea ref={ref} onPaste={stopPastePropagation} {...props} />;
});

export default memo(IAITextarea);
