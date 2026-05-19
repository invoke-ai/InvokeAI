import { Button, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { randomSeedChanged, selectDynamicPromptsMode } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ParamDynamicPromptsReshuffle = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const mode = useAppSelector(selectDynamicPromptsMode);

  const reshuffle = useCallback(() => {
    dispatch(randomSeedChanged(Date.now()));
  }, [dispatch]);

  if (mode !== 'random') {
    return null;
  }

  return (
    <FormControl>
      <FormLabel>{t('dynamicPrompts.preview')}</FormLabel>
      <Button onClick={reshuffle} variant="outline">
        {t('dynamicPrompts.reshuffleNow')}
      </Button>
    </FormControl>
  );
};

export default memo(ParamDynamicPromptsReshuffle);
