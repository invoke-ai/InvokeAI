import type { FlexProps, SystemStyleObject } from '@chakra-ui/react';
import { Box, Flex } from '@chakra-ui/react';
import type { CSSProperties } from 'react';
import { memo, useMemo } from 'react';
import { PanelResizeHandle } from 'react-resizable-panels';

type ResizeHandleProps = Omit<FlexProps, 'direction'> & {
  direction?: 'horizontal' | 'vertical';
  collapsedDirection?: 'top' | 'bottom' | 'left' | 'right';
  isCollapsed?: boolean;
};

const ResizeHandle = (props: ResizeHandleProps) => {
  const {
    direction = 'horizontal',
    collapsedDirection,
    isCollapsed = false,
    ...rest
  } = props;

  const resizeHandleStyles = useMemo<CSSProperties>(() => {
    if (direction === 'horizontal') {
      return {
        visibility: isCollapsed ? 'hidden' : 'visible',
        width: isCollapsed ? 0 : 'auto',
      };
    }
    return {
      visibility: isCollapsed ? 'hidden' : 'visible',
      width: isCollapsed ? 0 : 'auto',
    };
  }, [direction, isCollapsed]);

  const resizeHandleWrapperStyles = useMemo<SystemStyleObject>(() => {
    if (direction === 'horizontal') {
      return {
        w: collapsedDirection ? 2.5 : 4,
        h: 'full',
        justifyContent: collapsedDirection
          ? collapsedDirection === 'left'
            ? 'flex-start'
            : 'flex-end'
          : 'center',
        alignItems: 'center',
        div: {
          bg: 'base.850',
        },
        _hover: {
          div: { bg: 'base.700' },
        },
      };
    }
    return {
      w: 'full',
      h: collapsedDirection ? 2.5 : 4,
      alignItems: collapsedDirection
        ? collapsedDirection === 'top'
          ? 'flex-start'
          : 'flex-end'
        : 'center',
      justifyContent: 'center',
      div: {
        bg: 'base.850',
      },
      _hover: {
        div: { bg: 'base.700' },
      },
    };
  }, [collapsedDirection, direction]);
  const resizeInnerStyles = useMemo<SystemStyleObject>(() => {
    if (direction === 'horizontal') {
      return {
        w: 1,
        h: 'calc(100% - 1rem)',
        borderRadius: 'base',
        transitionProperty: 'common',
        transitionDuration: 'normal',
      };
    }

    return {
      h: 1,
      w: 'calc(100% - 1rem)',
      borderRadius: 'base',
      transitionProperty: 'common',
      transitionDuration: 'normal',
    };
  }, [direction]);

  return (
    <PanelResizeHandle style={resizeHandleStyles}>
      <Flex
        className="resize-handle-horizontal"
        sx={resizeHandleWrapperStyles}
        {...rest}
      >
        <Box sx={resizeInnerStyles} />
      </Flex>
    </PanelResizeHandle>
  );
};

export default memo(ResizeHandle);
