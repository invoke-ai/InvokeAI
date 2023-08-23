import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { FaLink } from 'react-icons/fa';
import { setShouldConcatSDXLStylePrompt } from '../store/sdxlSlice';

export default function ParamSDXLConcatButton() {
  const shouldConcatSDXLStylePrompt = useAppSelector(
    (state: RootState) => state.sdxl.shouldConcatSDXLStylePrompt
  );

  const dispatch = useAppDispatch();

  const handleShouldConcatPromptChange = () => {
    dispatch(setShouldConcatSDXLStylePrompt(!shouldConcatSDXLStylePrompt));
  };

  return (
    <IAIIconButton
      aria-label="Concatenate Prompt & Style"
      tooltip="Concatenate Prompt & Style"
      variant="outline"
      isChecked={shouldConcatSDXLStylePrompt}
      onClick={handleShouldConcatPromptChange}
      icon={<FaLink />}
      size="xs"
      sx={{
        position: 'absolute',
        insetInlineEnd: 1,
        top: 6,
        border: 'none',
        color: shouldConcatSDXLStylePrompt ? 'accent.500' : 'base.500',
        _hover: {
          bg: 'none',
        },
      }}
    ></IAIIconButton>
  );
}
