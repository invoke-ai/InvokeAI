import IAIMantineSelect, {
  IAISelectDataType,
  IAISelectProps,
} from 'common/components/IAIMantineSelect';
import { useTranslation } from 'react-i18next';

const variantSelectData: IAISelectDataType[] = [
  { value: 'normal', label: 'Normal' },
  { value: 'inpaint', label: 'Inpaint' },
  { value: 'depth', label: 'Depth' },
];

type VariantSelectProps = Omit<IAISelectProps, 'data'>;

export default function ModelVariantSelect(props: VariantSelectProps) {
  const { ...rest } = props;
  const { t } = useTranslation();

  return (
    <IAIMantineSelect
      label={t('modelManager.variant')}
      data={variantSelectData}
      {...rest}
    />
  );
}
