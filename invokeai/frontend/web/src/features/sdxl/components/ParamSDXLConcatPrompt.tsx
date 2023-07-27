import { Box } from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { ChangeEvent } from 'react';
import { setShouldConcatSDXLStylePrompt } from '../store/sdxlSlice';

export default function ParamSDXLConcatPrompt() {
  const shouldConcatSDXLStylePrompt = useAppSelector(
    (state: RootState) => state.sdxl.shouldConcatSDXLStylePrompt
  );

  const dispatch = useAppDispatch();

  const handleShouldConcatPromptChange = (e: ChangeEvent<HTMLInputElement>) => {
    dispatch(setShouldConcatSDXLStylePrompt(e.target.checked));
  };

  return (
    <Box
      sx={{
        px: 2,
      }}
    >
      <IAISwitch
        label="Concat Style Prompt"
        tooltip="Concatenates Basic Prompt with Style (Recommended)"
        isChecked={shouldConcatSDXLStylePrompt}
        onChange={handleShouldConcatPromptChange}
      />
    </Box>
  );
}
