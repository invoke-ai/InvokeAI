import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import type {
  InvSelectOnChange,
  InvSelectOption,
} from 'common/components/InvSelect/types';
import { setHrfMethod } from 'features/hrf/store/hrfSlice';
import { isParameterHRFMethod } from 'features/parameters/types/parameterSchemas';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const options: InvSelectOption[] = [
  { label: 'ESRGAN', value: 'ESRGAN' },
  { label: 'bilinear', value: 'bilinear' },
];

const ParamHrfMethodSelect = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const hrfMethod = useAppSelector((s) => s.hrf.hrfMethod);

  const onChange = useCallback<InvSelectOnChange>(
    (v) => {
      if (!isParameterHRFMethod(v?.value)) {
        return;
      }
      dispatch(setHrfMethod(v.value));
    },
    [dispatch]
  );

  const value = useMemo(
    () => options.find((o) => o.value === hrfMethod),
    [hrfMethod]
  );

  return (
    <InvControl label={t('hrf.upscaleMethod')}>
      <InvSelect value={value} options={options} onChange={onChange} />
    </InvControl>
  );
};

export default memo(ParamHrfMethodSelect);
