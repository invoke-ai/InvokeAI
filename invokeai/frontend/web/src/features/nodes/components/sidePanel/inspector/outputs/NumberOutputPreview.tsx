import { Text } from '@chakra-ui/react';
import { memo } from 'react';
import { FloatOutput, IntegerOutput } from 'services/api/types';

type Props = {
  output: IntegerOutput | FloatOutput;
};

const NumberOutputPreview = ({ output }: Props) => {
  return <Text>{output.value}</Text>;
};

export default memo(NumberOutputPreview);
