import { Box, FormControl, useDisclosure } from '@chakra-ui/react';
import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAITextarea from 'common/components/IAITextarea';
import AddEmbeddingButton from 'features/embedding/components/AddEmbeddingButton';
import ParamEmbeddingPopover from 'features/embedding/components/ParamEmbeddingPopover';
import { setNegativePrompt } from 'features/parameters/store/generationSlice';
import { ChangeEvent, KeyboardEvent, memo, useCallback, useRef } from 'react';
import { flushSync } from 'react-dom';
import { useTranslation } from 'react-i18next';
import { useFeatureStatus } from '../../../../system/hooks/useFeatureStatus';

const ParamNegativeConditioning = () => {
  const negativePrompt = useAppSelector(
    (state: RootState) => state.generation.negativePrompt
  );
  const promptRef = useRef<HTMLTextAreaElement>(null);
  const { isOpen, onClose, onOpen } = useDisclosure();
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChangePrompt = useCallback(
    (e: ChangeEvent<HTMLTextAreaElement>) => {
      dispatch(setNegativePrompt(e.target.value));
    },
    [dispatch]
  );
  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === '<') {
        onOpen();
      }
    },
    [onOpen]
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

      let newPrompt = negativePrompt.slice(0, caret);

      if (newPrompt[newPrompt.length - 1] !== '<') {
        newPrompt += '<';
      }

      newPrompt += `${v}>`;

      // we insert the cursor after the `>`
      const finalCaretPos = newPrompt.length;

      newPrompt += negativePrompt.slice(caret);

      // must flush dom updates else selection gets reset
      flushSync(() => {
        dispatch(setNegativePrompt(newPrompt));
      });

      // set the caret position to just after the TI trigger promptRef.current.selectionStart = finalCaretPos;
      promptRef.current.selectionEnd = finalCaretPos;
      onClose();
    },
    [dispatch, onClose, negativePrompt]
  );

  const isEmbeddingEnabled = useFeatureStatus('embedding').isFeatureEnabled;

  return (
    <FormControl>
      <ParamEmbeddingPopover
        isOpen={isOpen}
        onClose={onClose}
        onSelect={handleSelectEmbedding}
      >
        <IAITextarea
          id="negativePrompt"
          name="negativePrompt"
          ref={promptRef}
          value={negativePrompt}
          placeholder={t('parameters.negativePromptPlaceholder')}
          onChange={handleChangePrompt}
          resize="vertical"
          fontSize="sm"
          minH={16}
          {...(isEmbeddingEnabled && { onKeyDown: handleKeyDown })}
        />
      </ParamEmbeddingPopover>
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
    </FormControl>
  );
};

export default memo(ParamNegativeConditioning);
