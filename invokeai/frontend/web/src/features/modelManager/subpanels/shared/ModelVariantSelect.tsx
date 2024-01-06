import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import type {
  InvSelectOnChange,
  InvSelectOption,
} from 'common/components/InvSelect/types';
import { typedMemo } from 'common/util/typedMemo';
import { useCallback, useMemo } from 'react';
import type { UseControllerProps } from 'react-hook-form';
import { useController } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import type {
  CheckpointModelConfig,
  DiffusersModelConfig,
} from 'services/api/types';

const options: InvSelectOption[] = [
  { value: 'normal', label: 'Normal' },
  { value: 'inpaint', label: 'Inpaint' },
  { value: 'depth', label: 'Depth' },
];

const ModelVariantSelect = <
  T extends CheckpointModelConfig | DiffusersModelConfig,
>(
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
    <InvControl label={t('modelManager.variant')}>
      <InvSelect value={value} options={options} onChange={onChange} />
    </InvControl>
  );
};

export default typedMemo(ModelVariantSelect);
