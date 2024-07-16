import { Flex, Grid, GridItem, IconButton } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { memo, useCallback, useState } from 'react';
import {
  PiArrowDownBold,
  PiArrowDownLeftBold,
  PiArrowDownRightBold,
  PiArrowLeftBold,
  PiArrowRightBold,
  PiArrowUpBold,
  PiArrowUpLeftBold,
  PiArrowUpRightBold,
  PiSquareBold,
} from 'react-icons/pi';

type ResizeDirection =
  | 'up-left'
  | 'up'
  | 'up-right'
  | 'left'
  | 'center-out'
  | 'right'
  | 'down-left'
  | 'down'
  | 'down-right';

export const CanvasResizer = memo(() => {
  const bbox = useAppSelector((s) => s.canvasV2.bbox);
  const [resizeDirection, setResizeDirection] = useState<ResizeDirection>('center-out');

  const setDirUpLeft = useCallback(() => {
    setResizeDirection('up-left');
  }, []);

  const setDirUp = useCallback(() => {
    setResizeDirection('up');
  }, []);

  const setDirUpRight = useCallback(() => {
    setResizeDirection('up-right');
  }, []);

  const setDirLeft = useCallback(() => {
    setResizeDirection('left');
  }, []);

  const setDirCenterOut = useCallback(() => {
    setResizeDirection('center-out');
  }, []);

  const setDirRight = useCallback(() => {
    setResizeDirection('right');
  }, []);

  const setDirDownLeft = useCallback(() => {
    setResizeDirection('down-left');
  }, []);

  const setDirDown = useCallback(() => {
    setResizeDirection('down');
  }, []);

  const setDirDownRight = useCallback(() => {
    setResizeDirection('down-right');
  }, []);

  return (
    <Flex p={2}>
      <Grid gridTemplateRows="1fr 1fr 1fr" gridTemplateColumns="1fr 1fr 1fr" gap={2}>
        <GridItem>
          <IconButton
            onClick={setDirUpLeft}
            aria-label="up-left"
            icon={<PiArrowUpLeftBold />}
            variant={resizeDirection === 'up-left' ? 'solid' : 'ghost'}
          />
        </GridItem>
        <GridItem>
          <IconButton
            onClick={setDirUp}
            aria-label="up"
            icon={<PiArrowUpBold />}
            variant={resizeDirection === 'up' ? 'solid' : 'ghost'}
          />
        </GridItem>
        <GridItem>
          <IconButton
            onClick={setDirUpRight}
            aria-label="up-right"
            icon={<PiArrowUpRightBold />}
            variant={resizeDirection === 'up-right' ? 'solid' : 'ghost'}
          />
        </GridItem>
        <GridItem>
          <IconButton
            onClick={setDirLeft}
            aria-label="left"
            icon={<PiArrowLeftBold />}
            variant={resizeDirection === 'left' ? 'solid' : 'ghost'}
          />
        </GridItem>
        <GridItem>
          <IconButton
            onClick={setDirCenterOut}
            aria-label="center-out"
            icon={<PiSquareBold />}
            variant={resizeDirection === 'center-out' ? 'solid' : 'ghost'}
          />
        </GridItem>
        <GridItem>
          <IconButton
            onClick={setDirRight}
            aria-label="right"
            icon={<PiArrowRightBold />}
            variant={resizeDirection === 'right' ? 'solid' : 'ghost'}
          />
        </GridItem>
        <GridItem>
          <IconButton
            onClick={setDirDownLeft}
            aria-label="down-left"
            icon={<PiArrowDownLeftBold />}
            variant={resizeDirection === 'down-left' ? 'solid' : 'ghost'}
          />
        </GridItem>
        <GridItem>
          <IconButton
            onClick={setDirDown}
            aria-label="down"
            icon={<PiArrowDownBold />}
            variant={resizeDirection === 'down' ? 'solid' : 'ghost'}
          />
        </GridItem>
        <GridItem>
          <IconButton
            onClick={setDirDownRight}
            aria-label="down-right"
            icon={<PiArrowDownRightBold />}
            variant={resizeDirection === 'down-right' ? 'solid' : 'ghost'}
          />
        </GridItem>
      </Grid>
    </Flex>
  );
});

CanvasResizer.displayName = 'CanvasResizer';
