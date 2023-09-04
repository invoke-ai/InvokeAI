import { Box, FormControl, useDisclosure } from '@chakra-ui/react';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { ChangeEvent, KeyboardEvent, memo, useCallback, useRef } from 'react';

import { createSelector } from '@reduxjs/toolkit';
import { clampSymmetrySteps } from 'features/parameters/store/generationSlice';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';

import { userInvoked } from 'app/store/actions';
import IAITextarea from 'common/components/IAITextarea';
import { useIsReadyToInvoke } from 'common/hooks/useIsReadyToInvoke';
import AddEmbeddingButton from 'features/embedding/components/AddEmbeddingButton';
import ParamEmbeddingPopover from 'features/embedding/components/ParamEmbeddingPopover';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { AnimatePresence } from 'framer-motion';
import { isEqual } from 'lodash-es';
import { flushSync } from 'react-dom';
import { setNegativeStylePromptSDXL } from '../store/sdxlSlice';
import SDXLConcatLink from './SDXLConcatLink';

const promptInputSelector = createSelector(
  [stateSelector, activeTabNameSelector],
  ({ sdxl }, activeTabName) => {
    const { negativeStylePrompt, shouldConcatSDXLStylePrompt } = sdxl;

    return {
      prompt: negativeStylePrompt,
      shouldConcatSDXLStylePrompt,
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
const ParamSDXLNegativeStyleConditioning = () => {
  const dispatch = useAppDispatch();
  const isReady = useIsReadyToInvoke();
  const promptRef = useRef<HTMLTextAreaElement>(null);
  const { isOpen, onClose, onOpen } = useDisclosure();

  const { prompt, activeTabName, shouldConcatSDXLStylePrompt } =
    useAppSelector(promptInputSelector);

  const handleChangePrompt = useCallback(
    (e: ChangeEvent<HTMLTextAreaElement>) => {
      dispatch(setNegativeStylePromptSDXL(e.target.value));
    },
    [dispatch]
  );

  const handleSelectEmbedding = useCallback(
    (v: string) => {
      if (!promptRef.current) {
        return;
      }

      // this is where we insert the TI trigger
      const caret = promptRef.current.selectionStart;

      if (caret === undefined) {
        return;
      }

      let newPrompt = prompt.slice(0, caret);

      if (newPrompt[newPrompt.length - 1] !== '<') {
        newPrompt += '<';
      }

      newPrompt += `${v}>`;

      // we insert the cursor after the `>`
      const finalCaretPos = newPrompt.length;

      newPrompt += prompt.slice(caret);

      // must flush dom updates else selection gets reset
      flushSync(() => {
        dispatch(setNegativeStylePromptSDXL(newPrompt));
      });

      // set the caret position to just after the TI trigger
      promptRef.current.selectionStart = finalCaretPos;
      promptRef.current.selectionEnd = finalCaretPos;
      onClose();
    },
    [dispatch, onClose, prompt]
  );

  const isEmbeddingEnabled = useFeatureStatus('embedding').isFeatureEnabled;

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && e.shiftKey === false && isReady) {
        e.preventDefault();
        dispatch(clampSymmetrySteps());
        dispatch(userInvoked(activeTabName));
      }
      if (isEmbeddingEnabled && e.key === '<') {
        onOpen();
      }
    },
    [isReady, dispatch, activeTabName, onOpen, isEmbeddingEnabled]
  );

  // const handleSelect = (e: MouseEvent<HTMLTextAreaElement>) => {
  //   const target = e.target as HTMLTextAreaElement;
  // setCaret({ start: target.selectionStart, end: target.selectionEnd });
  // };

  return (
    <Box position="relative">
      <AnimatePresence>
        {shouldConcatSDXLStylePrompt && (
          <Box
            sx={{
              position: 'absolute',
              left: '3',
              w: '94%',
              top: '-17px',
            }}
          >
            <SDXLConcatLink />
          </Box>
        )}
      </AnimatePresence>
      <FormControl>
        <ParamEmbeddingPopover
          isOpen={isOpen}
          onClose={onClose}
          onSelect={handleSelectEmbedding}
        >
          <IAITextarea
            id="prompt"
            name="prompt"
            ref={promptRef}
            value={prompt}
            placeholder="Negative Style Prompt"
            onChange={handleChangePrompt}
            onKeyDown={handleKeyDown}
            resize="vertical"
            fontSize="sm"
            minH={16}
          />
        </ParamEmbeddingPopover>
      </FormControl>
      {!isOpen && isEmbeddingEnabled && (
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            insetInlineEnd: 0,
          }}
        >
          <AddEmbeddingButton onClick={onOpen} />
        </Box>
      )}
    </Box>
  );
};

export default memo(ParamSDXLNegativeStyleConditioning);
