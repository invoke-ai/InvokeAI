import { FormControl, Textarea } from '@chakra-ui/react';
import { ChangeEvent, KeyboardEvent, useRef } from 'react';
import { RootState, useAppDispatch, useAppSelector } from '../../../app/store';
import { generateImage } from '../../../app/socketio/actions';

import { OptionsState, setPrompt } from '../optionsSlice';
import { createSelector } from '@reduxjs/toolkit';
import { isEqual } from 'lodash';
import useCheckParameters, {
  systemSelector,
} from '../../../common/hooks/useCheckParameters';
import { useHotkeys } from 'react-hotkeys-hook';
import { tabMap } from '../../tabs/InvokeTabs';

export const optionsSelector = createSelector(
  (state: RootState) => state.options,
  (options: OptionsState) => {
    const { prompt, activeTab } = options;
    return {
      prompt,
      activeTabName: tabMap[activeTab],
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

/**
 * Prompt input text area.
 */
const PromptInput = () => {
  const promptRef = useRef<HTMLTextAreaElement>(null);
  const { prompt, activeTabName } = useAppSelector(optionsSelector);
  const { isProcessing } = useAppSelector(systemSelector);
  const dispatch = useAppDispatch();
  const isReady = useCheckParameters();

  const handleChangePrompt = (e: ChangeEvent<HTMLTextAreaElement>) => {
    dispatch(setPrompt(e.target.value));
  };

  useHotkeys(
    'ctrl+enter, cmd+enter',
    () => {
      if (isReady) {
        dispatch(generateImage(activeTabName));
      }
    },
    [isReady, activeTabName]
  );

  useHotkeys(
    'alt+a',
    () => {
      promptRef.current?.focus();
    },
    []
  );

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && e.shiftKey === false && isReady) {
      e.preventDefault();
      dispatch(generateImage(activeTabName));
    }
  };

  return (
    <div className="prompt-bar">
      <FormControl
        isInvalid={prompt.length === 0 || Boolean(prompt.match(/^[\s\r\n]+$/))}
      >
        <Textarea
          id="prompt"
          name="prompt"
          placeholder="I'm dreaming of..."
          size={'lg'}
          value={prompt}
          onChange={handleChangePrompt}
          onKeyDown={handleKeyDown}
          resize="vertical"
          height={30}
          ref={promptRef}
        />
      </FormControl>
    </div>
  );
};

export default PromptInput;
