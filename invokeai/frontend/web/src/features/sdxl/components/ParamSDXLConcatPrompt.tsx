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
        px: 4,
        py: 2,
        borderRadius: 4,
        bg: 'base.100',
        _dark: { bg: 'base.800' },
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
