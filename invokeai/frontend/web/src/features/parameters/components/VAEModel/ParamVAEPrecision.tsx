import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import type { InvSelectOnChange } from 'common/components/InvSelect/types';
import { vaePrecisionChanged } from 'features/parameters/store/generationSlice';
import { isParameterPrecision } from 'features/parameters/types/parameterSchemas';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const options = [
  { label: 'FP16', value: 'fp16' },
  { label: 'FP32', value: 'fp32' },
];

const ParamVAEModelSelect = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const vaePrecision = useAppSelector((s) => s.generation.vaePrecision);

  const onChange = useCallback<InvSelectOnChange>(
    (v) => {
      if (!isParameterPrecision(v?.value)) {
        return;
      }

      dispatch(vaePrecisionChanged(v.value));
    },
    [dispatch]
  );

  const value = useMemo(
    () => options.find((o) => o.value === vaePrecision),
    [vaePrecision]
  );

  return (
    <InvControl
      label={t('modelManager.vaePrecision')}
      feature="paramVAEPrecision"
      w="14rem"
      flexShrink={0}
    >
      <InvSelect value={value} options={options} onChange={onChange} />
    </InvControl>
  );
};

export default memo(ParamVAEModelSelect);
