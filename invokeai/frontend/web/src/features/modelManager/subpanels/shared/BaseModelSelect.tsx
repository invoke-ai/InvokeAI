import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import type {
  InvSelectOnChange,
  InvSelectOption,
} from 'common/components/InvSelect/types';
import { typedMemo } from 'common/util/typedMemo';
import { MODEL_TYPE_MAP } from 'features/parameters/types/constants';
import { useCallback, useMemo } from 'react';
import type { UseControllerProps } from 'react-hook-form';
import { useController } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import type { AnyModelConfig } from 'services/api/types';

const options: InvSelectOption[] = [
  { value: 'sd-1', label: MODEL_TYPE_MAP['sd-1'] },
  { value: 'sd-2', label: MODEL_TYPE_MAP['sd-2'] },
  { value: 'sdxl', label: MODEL_TYPE_MAP['sdxl'] },
  { value: 'sdxl-refiner', label: MODEL_TYPE_MAP['sdxl-refiner'] },
];

const BaseModelSelect = <T extends AnyModelConfig>(
  props: UseControllerProps<T>
) => {
  const { t } = useTranslation();
  const { field } = useController(props);
  const value = useMemo(
    () => options.find((o) => o.value === field.value),
    [field.value]
  );
  const onChange = useCallback<InvSelectOnChange>(
    (v) => {
      field.onChange(v?.value);
    },
    [field]
  );
  return (
    <InvControl label={t('modelManager.baseModel')}>
      <InvSelect value={value} options={options} onChange={onChange} />
    </InvControl>
  );
};

export default typedMemo(BaseModelSelect);
