import { useCallback, useEffect, useRef, useState } from 'react';
import { Box, Flex } from '@invoke-ai/ui-library';
import GalleryImageGrid from './GalleryImageGrid';
import { useAppSelector, useAppDispatch } from '../../../../app/store/storeHooks';
import { limitChanged } from '../../store/gallerySlice';

export const GalleryImageGridContainer = () => {
  const { galleryImageMinimumWidth, limit } = useAppSelector((s) => s.gallery);
  const dispatch = useAppDispatch();
  const containerRef = useRef<HTMLDivElement>(null);

  const calculateItemsPerPage = useCallback(() => {
    const containerWidth = containerRef.current?.clientWidth;
    const containerHeight = containerRef.current?.clientHeight;
    console.log({ containerWidth, containerHeight, galleryImageMinimumWidth });
    if (containerHeight && containerWidth) {
      const numberHorizontal = Math.floor(containerWidth / galleryImageMinimumWidth);
      const imageWidth = containerWidth / numberHorizontal;
      const numberAllowedVertical = Math.floor(containerHeight / imageWidth);
      console.log({ numberAllowedVertical, numberHorizontal });
      dispatch(limitChanged(numberAllowedVertical * numberHorizontal));
    }
  }, [containerRef, galleryImageMinimumWidth]);

  useEffect(() => {
    dispatch(limitChanged(undefined));
    calculateItemsPerPage();
  }, [galleryImageMinimumWidth]);

  useEffect(() => {
    if (!containerRef.current) {
      return;
    }

    const resizeObserver = new ResizeObserver(() => {
      console.log('resize');
      if (!containerRef.current) {
        return;
      }
      dispatch(limitChanged(undefined));
      calculateItemsPerPage();
    });

    resizeObserver.observe(containerRef.current);
    dispatch(limitChanged(undefined));
    calculateItemsPerPage();

    return () => {
      resizeObserver.disconnect();
    };
  }, []);

  return (
    <Flex flexDir="column" w="full" h="full" overflow="hidden" ref={containerRef}>
      {limit && <GalleryImageGrid />}
    </Flex>
  );
};
