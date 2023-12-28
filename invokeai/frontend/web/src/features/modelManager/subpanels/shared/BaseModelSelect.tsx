import { InvSelect } from 'common/components/InvSelect/InvSelect';
import type {
  InvSelectOption,
  InvSelectProps,
} from 'common/components/InvSelect/types';
import { MODEL_TYPE_MAP } from 'features/parameters/types/constants';

const options: InvSelectOption[] = [
  { value: 'sd-1', label: MODEL_TYPE_MAP['sd-1'] },
  { value: 'sd-2', label: MODEL_TYPE_MAP['sd-2'] },
  { value: 'sdxl', label: MODEL_TYPE_MAP['sdxl'] },
  { value: 'sdxl-refiner', label: MODEL_TYPE_MAP['sdxl-refiner'] },
];

type BaseModelSelectProps = Omit<InvSelectProps, 'options'>;

export default function BaseModelSelect(props: BaseModelSelectProps) {
  return <InvSelect options={options} {...props} />;
}
