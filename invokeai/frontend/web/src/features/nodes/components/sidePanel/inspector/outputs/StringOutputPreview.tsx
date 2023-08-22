import { Text } from '@chakra-ui/react';
import { memo } from 'react';
import { StringOutput } from 'services/api/types';

type Props = {
  output: StringOutput;
};

const StringOutputPreview = ({ output }: Props) => {
  return <Text>{output.value}</Text>;
};

export default memo(StringOutputPreview);
