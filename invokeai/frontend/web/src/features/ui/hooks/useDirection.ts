import { useTheme } from '@chakra-ui/react';
import { useMemo } from 'react';
import { LangDirection } from '../components/common/ResizableDrawer/types';

export const useLangDirection = () => {
  const theme = useTheme();

  const langDirection = useMemo(
    () => theme.direction as LangDirection,
    [theme.direction]
  );

  return langDirection;
};
