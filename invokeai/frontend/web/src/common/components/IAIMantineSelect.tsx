import { Select, SelectProps } from '@mantine/core';
import { memo } from 'react';

type IAISelectProps = SelectProps;

const IAIMantineSelect = (props: IAISelectProps) => {
  const { searchable = true, ...rest } = props;
  return <Select searchable={searchable} {...rest} />;
};

export default memo(IAIMantineSelect);
