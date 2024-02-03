import { FormControl, FormLabel, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { combinatorialToggled } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamDynamicPromptsCombinatorial = () => {
  const combinatorial = useAppSelector((s) => s.dynamicPrompts.combinatorial);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback(() => {
    dispatch(combinatorialToggled());
  }, [dispatch]);

  return (
    <FormControl>
      <FormLabel>{t('dynamicPrompts.combinatorial')}</FormLabel>
      <Switch isChecked={combinatorial} onChange={handleChange} />
    </FormControl>
  );
};

export default memo(ParamDynamicPromptsCombinatorial);
