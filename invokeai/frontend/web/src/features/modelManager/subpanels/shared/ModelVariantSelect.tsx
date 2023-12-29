import { InvControl } from 'common/components/InvControl/InvControl';
import { InvSelect } from 'common/components/InvSelect/InvSelect';
import type {
  InvSelectOption,
  InvSelectProps,
} from 'common/components/InvSelect/types';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const options: InvSelectOption[] = [
  { value: 'normal', label: 'Normal' },
  { value: 'inpaint', label: 'Inpaint' },
  { value: 'depth', label: 'Depth' },
];

type VariantSelectProps = Omit<InvSelectProps, 'options'>;

const ModelVariantSelect = (props: VariantSelectProps) => {
  const { t } = useTranslation();
  return (
    <InvControl label={t('modelManager.variant')}>
      <InvSelect options={options} {...props} />
    </InvControl>
  );
};

export default memo(ModelVariantSelect);
