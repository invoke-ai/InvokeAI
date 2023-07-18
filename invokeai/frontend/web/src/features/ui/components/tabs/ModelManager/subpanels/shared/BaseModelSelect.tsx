import IAIMantineSelect, {
  IAISelectDataType,
  IAISelectProps,
} from 'common/components/IAIMantineSelect';
import { MODEL_TYPE_MAP } from 'features/parameters/types/constants';
import { useTranslation } from 'react-i18next';

const baseModelSelectData: IAISelectDataType[] = [
  { value: 'sd-1', label: MODEL_TYPE_MAP['sd-1'] },
  { value: 'sd-2', label: MODEL_TYPE_MAP['sd-2'] },
  { value: 'sdxl', label: MODEL_TYPE_MAP['sdxl'] },
  { value: 'sdxl-refiner', label: MODEL_TYPE_MAP['sdxl-refiner'] },
];

type BaseModelSelectProps = Omit<IAISelectProps, 'data'>;

export default function BaseModelSelect(props: BaseModelSelectProps) {
  const { ...rest } = props;
  const { t } = useTranslation();

  return (
    <IAIMantineSelect
      label={t('modelManager.baseModel')}
      data={baseModelSelectData}
      {...rest}
    />
  );
}
