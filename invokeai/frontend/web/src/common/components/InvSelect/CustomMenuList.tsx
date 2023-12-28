import { Box } from '@chakra-ui/layout';
import type { GroupBase, MenuListProps } from 'chakra-react-select';
import { chakraComponents } from 'chakra-react-select';
import { overlayScrollbarsParams } from 'common/components/OverlayScrollbars/constants';
import { cloneDeep, merge } from 'lodash-es';
import type { UseOverlayScrollbarsParams } from 'overlayscrollbars-react';
import { useOverlayScrollbars } from 'overlayscrollbars-react';
import type { PropsWithChildren } from 'react';
import { useEffect, useRef, useState } from 'react';

import type { InvSelectOption } from './types';

type CustomMenuListProps = MenuListProps<
  InvSelectOption,
  false,
  GroupBase<InvSelectOption>
>;

const overlayScrollbarsParamsOverrides: Partial<UseOverlayScrollbarsParams> = {
  options: { scrollbars: { autoHide: 'never' } },
};

const osParams = merge(
  cloneDeep(overlayScrollbarsParams),
  overlayScrollbarsParamsOverrides
);

const Scrollable = (
  props: PropsWithChildren<{ viewport: HTMLDivElement | null }>
) => {
  const { children, viewport } = props;

  const targetRef = useRef<HTMLDivElement>(null);
  const [initialize, getInstance] = useOverlayScrollbars(osParams);

  useEffect(() => {
    if (targetRef.current && viewport) {
      initialize({
        target: targetRef.current,
        elements: {
          viewport,
        },
      });
    }
    return () => getInstance()?.destroy();
  }, [viewport, initialize, getInstance]);

  return (
    <Box
      ref={targetRef}
      data-overlayscrollbars=""
      border="none"
      shadow="dark-lg"
      borderRadius="md"
      p={1}
    >
      {children}
    </Box>
  );
};

export const CustomMenuList = ({
  children,
  innerRef,
  ...other
}: CustomMenuListProps) => {
  const [viewport, setViewport] = useState<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!innerRef || !(innerRef instanceof Function)) {
      return;
    }
    innerRef(viewport);
  }, [innerRef, viewport]);

  return (
    <Scrollable viewport={viewport}>
      <chakraComponents.MenuList {...other} innerRef={setViewport}>
        {children}
      </chakraComponents.MenuList>
    </Scrollable>
  );
};
