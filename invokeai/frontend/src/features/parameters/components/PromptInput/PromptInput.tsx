import { FormControl, Textarea } from '@chakra-ui/react';
import { generateImage } from 'app/socketio/actions';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import { ChangeEvent, KeyboardEvent, useRef } from 'react';

import { createSelector } from '@reduxjs/toolkit';
import { readinessSelector } from 'app/selectors/readinessSelector';
import {
  GenerationState,
  setPrompt,
} from 'features/parameters/store/generationSlice';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';

import { isEqual } from 'lodash';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';

const promptInputSelector = createSelector(
  [(state: RootState) => state.generation, activeTabNameSelector],
  (parameters: GenerationState, activeTabName) => {
    return {
      prompt: parameters.prompt,
      activeTabName,
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
  const dispatch = useAppDispatch();
  const { prompt, activeTabName } = useAppSelector(promptInputSelector);
  const { isReady } = useAppSelector(readinessSelector);

  const promptRef = useRef<HTMLTextAreaElement>(null);

  const { t } = useTranslation();

  const handleChangePrompt = (e: ChangeEvent<HTMLTextAreaElement>) => {
    dispatch(setPrompt(e.target.value));
  };

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
          placeholder={t('parameters.promptPlaceholder')}
          size="lg"
          value={prompt}
          onChange={handleChangePrompt}
          onKeyDown={handleKeyDown}
          resize="vertical"
          height={30}
          ref={promptRef}
          _placeholder={{
            color: 'var(--text-color-secondary)',
          }}
        />
      </FormControl>
    </div>
  );
};

export default PromptInput;
