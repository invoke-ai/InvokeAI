import { createSelector } from '@reduxjs/toolkit';
// import IAICanvas from 'features/canvas/components/IAICanvas';
import IAICanvasResizer from 'features/canvas/components/IAICanvasResizer';
import _ from 'lodash';
import { useLayoutEffect } from 'react';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import { setDoesCanvasNeedScaling } from 'features/canvas/store/canvasSlice';
import IAICanvas from 'features/canvas/components/IAICanvas';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import { Flex } from '@chakra-ui/react';
import UnifiedCanvasToolbarBeta from './UnifiedCanvasToolbarBeta';
import UnifiedCanvasToolSettingsBeta from './UnifiedCanvasToolSettingsBeta';

const selector = createSelector(
  [canvasSelector],
  (canvas) => {
    const { doesCanvasNeedScaling } = canvas;
    return {
      doesCanvasNeedScaling,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

const UnifiedCanvasDisplayBeta = () => {
  const dispatch = useAppDispatch();

  const { doesCanvasNeedScaling } = useAppSelector(selector);

  useLayoutEffect(() => {
    dispatch(setDoesCanvasNeedScaling(true));

    const resizeCallback = _.debounce(() => {
      dispatch(setDoesCanvasNeedScaling(true));
    }, 250);

    window.addEventListener('resize', resizeCallback);

    return () => window.removeEventListener('resize', resizeCallback);
  }, [dispatch]);

  return (
    <div className={'workarea-single-view'}>
      <Flex
        flexDirection={'row'}
        width="100%"
        height="100%"
        columnGap={'1rem'}
        padding="1rem"
      >
        <UnifiedCanvasToolbarBeta />
        <Flex
          width="100%"
          height="100%"
          flexDirection={'column'}
          rowGap={'1rem'}
        >
          <UnifiedCanvasToolSettingsBeta />
          {doesCanvasNeedScaling ? <IAICanvasResizer /> : <IAICanvas />}
        </Flex>
      </Flex>
    </div>
  );
};

export default UnifiedCanvasDisplayBeta;
