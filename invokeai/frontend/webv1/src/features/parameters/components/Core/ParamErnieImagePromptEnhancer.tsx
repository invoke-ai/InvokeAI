import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectErnieImageUsePromptEnhancer,
  setErnieImageUsePromptEnhancer,
} from 'features/controlLayers/store/paramsSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamErnieImagePromptEnhancer = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const useEnhancer = useAppSelector(selectErnieImageUsePromptEnhancer);

  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(setErnieImageUsePromptEnhancer(e.target.checked));
    },
    [dispatch]
  );

  return (
    <FormControl>
      <FormLabel>{t('parameters.ernieImagePromptEnhancer')}</FormLabel>
      <Switch isChecked={useEnhancer} onChange={onChange} />
    </FormControl>
  );
};

export default memo(ParamErnieImagePromptEnhancer);
