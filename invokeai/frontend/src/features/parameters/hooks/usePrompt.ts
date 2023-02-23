import { getPromptAndNegative } from 'common/util/getPromptAndNegative';

import * as InvokeAI from 'app/invokeai';
import promptToString from 'common/util/promptToString';
import { useAppDispatch } from 'app/storeHooks';
import { setNegativePrompt, setPrompt } from '../store/generationSlice';

// TECHDEBT: We have two metadata prompt formats and need to handle recalling either of them.
// This hook provides a function to do that.
const useSetBothPrompts = () => {
  const dispatch = useAppDispatch();

  return (inputPrompt: InvokeAI.Prompt) => {
    const promptString =
      typeof inputPrompt === 'string'
        ? inputPrompt
        : promptToString(inputPrompt);

    const [prompt, negativePrompt] = getPromptAndNegative(promptString);

    dispatch(setPrompt(prompt));
    dispatch(setNegativePrompt(negativePrompt));
  };
};

export default useSetBothPrompts;
