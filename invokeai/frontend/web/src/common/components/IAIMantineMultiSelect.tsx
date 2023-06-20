import { Tooltip } from '@chakra-ui/react';
import { MultiSelect, MultiSelectProps } from '@mantine/core';
import { memo } from 'react';

type IAIMultiSelectProps = MultiSelectProps & {
  tooltip?: string;
};

const IAIMantineMultiSelect = (props: IAIMultiSelectProps) => {
  const { searchable = true, tooltip, ...rest } = props;
  return (
    <Tooltip label={tooltip} placement="top" hasArrow>
      <MultiSelect
        searchable={searchable}
        styles={() => ({
          label: {
            color: 'var(--invokeai-colors-base-300)',
            fontWeight: 'normal',
          },
          searchInput: {
            '::placeholder': {
              color: 'var(--invokeai-colors-base-700)',
            },
          },
          input: {
            backgroundColor: 'var(--invokeai-colors-base-900)',
            borderWidth: '2px',
            borderColor: 'var(--invokeai-colors-base-800)',
            color: 'var(--invokeai-colors-base-100)',
            padding: 10,
            paddingRight: 24,
            fontWeight: 600,
            '&:hover': { borderColor: 'var(--invokeai-colors-base-700)' },
            '&:focus': {
              borderColor: 'var(--invokeai-colors-accent-600)',
            },
            '&:focus-within': {
              borderColor: 'var(--invokeai-colors-accent-600)',
            },
          },
          value: {
            backgroundColor: 'var(--invokeai-colors-base-800)',
            color: 'var(--invokeai-colors-base-100)',
            button: {
              color: 'var(--invokeai-colors-base-100)',
            },
            '&:hover': {
              backgroundColor: 'var(--invokeai-colors-base-700)',
              cursor: 'pointer',
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
            padding: 20,
            button: {
              color: 'var(--invokeai-colors-base-100)',
            },
          },
        })}
        {...rest}
      />
    </Tooltip>
  );
};

export default memo(IAIMantineMultiSelect);
