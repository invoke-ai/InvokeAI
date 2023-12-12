import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIInformationalPopover from 'common/components/IAIInformationalPopover/IAIInformationalPopover';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { setInfillMethod } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetAppConfigQuery } from 'services/api/endpoints/appInfo';

const selector = createMemoizedSelector([stateSelector], ({ generation }) => {
  const { infillMethod } = generation;

  return {
    infillMethod,
  };
});

const ParamInfillMethod = () => {
  const dispatch = useAppDispatch();
  const { infillMethod } = useAppSelector(selector);

  const { data: appConfigData, isLoading } = useGetAppConfigQuery();

  const infill_methods = appConfigData?.infill_methods;

  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: string) => {
      dispatch(setInfillMethod(v));
    },
    [dispatch]
  );

  return (
    <IAIInformationalPopover feature="infillMethod">
      <IAIMantineSelect
        disabled={infill_methods?.length === 0}
        placeholder={isLoading ? 'Loading...' : undefined}
        label={t('parameters.infillMethod')}
        value={infillMethod}
        data={infill_methods ?? []}
        onChange={handleChange}
      />
    </IAIInformationalPopover>
  );
};

export default memo(ParamInfillMethod);
