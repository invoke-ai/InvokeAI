import { useToken } from '@chakra-ui/react';
import { ReactNode } from 'react';

type IAIOptionProps = {
  children: ReactNode | string | number;
  value: string | number;
  key: string | number;
};

export default function IAIOption(props: IAIOptionProps) {
  const { children, value, key } = props;
  const [base800, base200] = useToken('colors', ['base.800', 'base.200']);

  return (
    <option
      key={key}
      value={value}
      style={{ background: base800, color: base200 }}
    >
      {children}
    </option>
  );
}
