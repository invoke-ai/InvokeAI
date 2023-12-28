import { InvSelect } from 'common/components/InvSelect/InvSelect';
import type {
  InvSelectOption,
  InvSelectProps,
} from 'common/components/InvSelect/types';

const options: InvSelectOption[] = [
  { value: 'normal', label: 'Normal' },
  { value: 'inpaint', label: 'Inpaint' },
  { value: 'depth', label: 'Depth' },
];

type VariantSelectProps = Omit<InvSelectProps, 'options'>;

export default function ModelVariantSelect(props: VariantSelectProps) {
  return <InvSelect options={options} {...props} />;
}
