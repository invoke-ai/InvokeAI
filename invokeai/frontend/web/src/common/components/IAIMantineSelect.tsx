import { Select, SelectProps } from '@mantine/core';
import { memo } from 'react';

type IAISelectProps = SelectProps;

const IAIMantineSelect = (props: IAISelectProps) => {
  const { searchable = true, ...rest } = props;
  return (
    <Select
      searchable={searchable}
      styles={() => ({
        label: {
          color: 'var(--invokeai-colors-base-300)',
          fontWeight: 'normal',
        },
        input: {
          backgroundColor: 'var(--invokeai-colors-base-900)',
          border: 'none',
          color: 'var(--invokeai-colors-base-100)',
          fontWeight: 500,
        },
        dropdown: {
          backgroundColor: 'var(--invokeai-colors-base-800)',
          borderColor: 'var(--invokeai-colors-base-700)',
        },
        item: {
          color: 'var(--invokeai-colors-base-300)',
          ':hover': {
            color: 'var(--invokeai-colors-base-300)',
            backgroundColor: 'var(--invokeai-colors-accent-750)',
          },
          '&[data-selected]': {
            color: 'var(--invokeai-colors-base-300)',
            backgroundColor: 'var(--invokeai-colors-accent-750)',
            fontWeight: 500,
            ':hover': {
              backgroundColor: 'var(--invokeai-colors-accent-750)',
            },
          },
        },
      })}
      {...rest}
    />
  );
};

export default memo(IAIMantineSelect);
