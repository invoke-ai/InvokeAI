import { Tooltip } from '@chakra-ui/react';
import { Select, SelectProps } from '@mantine/core';
import { memo } from 'react';

export type IAISelectDataType = {
  value: string;
  label: string;
  tooltip?: string;
};

type IAISelectProps = SelectProps & {
  tooltip?: string;
};

const IAIMantineSelect = (props: IAISelectProps) => {
  const { searchable = true, tooltip, ...rest } = props;
  return (
    <Tooltip label={tooltip} placement="top" hasArrow>
      <Select
        searchable={searchable}
        styles={() => ({
          label: {
            color: 'var(--invokeai-colors-base-300)',
            fontWeight: 'normal',
          },
          input: {
            backgroundColor: 'var(--invokeai-colors-base-900)',
            borderWidth: '2px',
            borderColor: 'var(--invokeai-colors-base-800)',
            color: 'var(--invokeai-colors-base-100)',
            paddingRight: 24,
            fontWeight: 600,
            '&:hover': { borderColor: 'var(--invokeai-colors-base-700)' },
            '&:focus': {
              borderColor: 'var(--invokeai-colors-accent-600)',
            },
          },
          dropdown: {
            backgroundColor: 'var(--invokeai-colors-base-800)',
            borderColor: 'var(--invokeai-colors-base-700)',
          },
          item: {
            backgroundColor: 'var(--invokeai-colors-base-800)',
            color: 'var(--invokeai-colors-base-200)',
            padding: 6,
            '&[data-hovered]': {
              color: 'var(--invokeai-colors-base-100)',
              backgroundColor: 'var(--invokeai-colors-base-750)',
            },
            '&[data-active]': {
              backgroundColor: 'var(--invokeai-colors-base-750)',
              '&:hover': {
                color: 'var(--invokeai-colors-base-100)',
                backgroundColor: 'var(--invokeai-colors-base-750)',
              },
            },
            '&[data-selected]': {
              color: 'var(--invokeai-colors-base-50)',
              backgroundColor: 'var(--invokeai-colors-accent-650)',
              fontWeight: 600,
              '&:hover': {
                backgroundColor: 'var(--invokeai-colors-accent-600)',
              },
            },
          },
          rightSection: {
            width: 24,
          },
        })}
        {...rest}
      />
    </Tooltip>
  );
};

export default memo(IAIMantineSelect);
