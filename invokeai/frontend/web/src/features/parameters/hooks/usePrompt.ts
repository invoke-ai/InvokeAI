import { getPromptAndNegative } from 'common/util/getPromptAndNegative';

import * as InvokeAI from 'app/types/invokeai';
import promptToString from 'common/util/promptToString';
import { useAppDispatch } from 'app/store/storeHooks';
import { setNegativePrompt, setPositivePrompt } from '../store/generationSlice';
import { useCallback } from 'react';

// TECHDEBT: We have two metadata prompt formats and need to handle recalling either of them.
// This hook provides a function to do that.
const useSetBothPrompts = () => {
  const dispatch = useAppDispatch();

  return useCallback(
    (inputPrompt: InvokeAI.Prompt, negativePrompt?: InvokeAI.Prompt) => {
      dispatch(setPositivePrompt(inputPrompt));
      if (negativePrompt) {
        dispatch(setNegativePrompt(negativePrompt));
      }
    },
    [dispatch]
  );
};

export default useSetBothPrompts;
