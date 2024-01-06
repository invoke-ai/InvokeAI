import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSwitch } from 'common/components/InvSwitch/wrapper';
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
    <InvControl label={t('dynamicPrompts.combinatorial')}>
      <InvSwitch isChecked={combinatorial} onChange={handleChange} />
    </InvControl>
  );
};

export default memo(ParamDynamicPromptsCombinatorial);
