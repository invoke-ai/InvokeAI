import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { combinatorialToggled } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector(stateSelector, (state) => {
  const { combinatorial } = state.dynamicPrompts;

  return { combinatorial };
});

const ParamDynamicPromptsCombinatorial = () => {
  const { combinatorial } = useAppSelector(selector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback(() => {
    dispatch(combinatorialToggled());
  }, [dispatch]);

  return (
    <IAISwitch
      label={t('dynamicPrompts.combinatorial')}
      isChecked={combinatorial}
      onChange={handleChange}
    />
  );
};

export default memo(ParamDynamicPromptsCombinatorial);
