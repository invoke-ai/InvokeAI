import { Box } from '@chakra-ui/react';

interface SubItemHookProps {
  active?: boolean;
  width?: string | number;
  height?: string | number;
  side?: 'left' | 'right';
}

export default function SubItemHook(props: SubItemHookProps) {
  const {
    active = true,
    width = '1rem',
    height = '1.3rem',
    side = 'right',
  } = props;
  return (
    <>
      {side === 'right' && (
        <Box
          width={width}
          height={height}
          margin="-0.5rem 0.5rem 0 0.5rem"
          borderLeft={
            active
              ? '3px solid var(--subhook-color)'
              : '3px solid var(--tab-hover-color)'
          }
          borderBottom={
            active
              ? '3px solid var(--subhook-color)'
              : '3px solid var(--tab-hover-color)'
          }
        />
      )}
      {side === 'left' && (
        <Box
          width={width}
          height={height}
          margin="-0.5rem 0.5rem 0 0.5rem"
          borderRight={
            active
              ? '3px solid var(--subhook-color)'
              : '3px solid var(--tab-hover-color)'
          }
          borderBottom={
            active
              ? '3px solid var(--subhook-color)'
              : '3px solid var(--tab-hover-color)'
          }
        />
      )}
    </>
  );
}
