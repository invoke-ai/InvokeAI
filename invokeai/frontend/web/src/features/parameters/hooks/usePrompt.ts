import { getPromptAndNegative } from 'common/util/getPromptAndNegative';

import * as InvokeAI from 'app/types/invokeai';
import promptToString from 'common/util/promptToString';
import { useAppDispatch } from 'app/store/storeHooks';
import { setNegativePrompt, setPrompt } from '../store/generationSlice';
import { useCallback } from 'react';

// TECHDEBT: We have two metadata prompt formats and need to handle recalling either of them.
// This hook provides a function to do that.
const useSetBothPrompts = () => {
  const dispatch = useAppDispatch();

  return useCallback(
    (inputPrompt: InvokeAI.Prompt) => {
      const promptString =
        typeof inputPrompt === 'string'
          ? inputPrompt
          : promptToString(inputPrompt);

      const [prompt, negativePrompt] = getPromptAndNegative(promptString);

      dispatch(setPrompt(prompt));
      dispatch(setNegativePrompt(negativePrompt));
    },
    [dispatch]
  );
};

export default useSetBothPrompts;
