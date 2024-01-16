import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import type {
  InvSelectOnChange,
  InvSelectOption,
} from 'common/components/InvSelect/types';
import { setMaskBlurMethod } from 'features/parameters/store/generationSlice';
import { isParameterMaskBlurMethod } from 'features/parameters/types/parameterSchemas';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const options: InvSelectOption[] = [
  { label: 'Box Blur', value: 'box' },
  { label: 'Gaussian Blur', value: 'gaussian' },
];

const ParamMaskBlurMethod = () => {
  const maskBlurMethod = useAppSelector((s) => s.generation.maskBlurMethod);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const onChange = useCallback<InvSelectOnChange>(
    (v) => {
      if (!isParameterMaskBlurMethod(v?.value)) {
        return;
      }
      dispatch(setMaskBlurMethod(v.value));
    },
    [dispatch]
  );

  const value = useMemo(
    () => options.find((o) => o.value === maskBlurMethod),
    [maskBlurMethod]
  );

  return (
    <InvControl
      label={t('parameters.maskBlurMethod')}
      feature="compositingBlurMethod"
    >
      <InvSelect value={value} onChange={onChange} options={options} />
    </InvControl>
  );
};

export default memo(ParamMaskBlurMethod);
