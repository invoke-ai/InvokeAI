import { useToken } from '@chakra-ui/react';
import { ReactNode } from 'react';

type IAIOptionProps = {
  children: ReactNode | string | number;
  value: string | number;
};

export default function IAIOption(props: IAIOptionProps) {
  const { children, value } = props;
  const [base800, base200] = useToken('colors', ['base.800', 'base.200']);

  return (
    <option value={value} style={{ background: base800, color: base200 }}>
      {children}
    </option>
  );
}
