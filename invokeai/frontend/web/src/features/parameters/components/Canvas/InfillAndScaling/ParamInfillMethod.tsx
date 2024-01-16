import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import type {
  InvSelectOnChange,
  InvSelectOption,
} from 'common/components/InvSelect/types';
import { setInfillMethod } from 'features/parameters/store/generationSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useGetAppConfigQuery } from 'services/api/endpoints/appInfo';

const ParamInfillMethod = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const infillMethod = useAppSelector((s) => s.generation.infillMethod);
  const { data: appConfigData } = useGetAppConfigQuery();
  const options = useMemo<InvSelectOption[]>(
    () =>
      appConfigData
        ? appConfigData.infill_methods.map((method) => ({
            label: method,
            value: method,
          }))
        : [],
    [appConfigData]
  );

  const onChange = useCallback<InvSelectOnChange>(
    (v) => {
      if (!v || !options.find((o) => o.value === v.value)) {
        return;
      }
      dispatch(setInfillMethod(v.value));
    },
    [dispatch, options]
  );

  const value = useMemo(
    () => options.find((o) => o.value === infillMethod),
    [options, infillMethod]
  );

  return (
    <InvControl
      label={t('parameters.infillMethod')}
      isDisabled={options.length === 0}
      feature="infillMethod"
    >
      <InvSelect value={value} options={options} onChange={onChange} />
    </InvControl>
  );
};

export default memo(ParamInfillMethod);
