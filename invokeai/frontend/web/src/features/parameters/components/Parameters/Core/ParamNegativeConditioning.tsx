import { FormControl } from '@chakra-ui/react';
import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAITextarea from 'common/components/IAITextarea';
import { setNegativePrompt } from 'features/parameters/store/generationSlice';
import { useTranslation } from 'react-i18next';

const ParamNegativeConditioning = () => {
  const negativePrompt = useAppSelector(
    (state: RootState) => state.generation.negativePrompt
  );

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  return (
    <FormControl>
      <IAITextarea
        id="negativePrompt"
        name="negativePrompt"
        value={negativePrompt}
        onChange={(e) => dispatch(setNegativePrompt(e.target.value))}
        placeholder={t('parameters.negativePrompts')}
        _focusVisible={{
          borderColor: 'error.600',
        }}
        fontSize="sm"
      />
    </FormControl>
  );
};

export default ParamNegativeConditioning;
