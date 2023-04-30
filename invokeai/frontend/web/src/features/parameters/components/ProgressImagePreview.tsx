import { Flex, Icon, Image, Text } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { systemSelector } from 'features/system/store/systemSelectors';
import { memo } from 'react';
import { FaImage, FaStopwatch } from 'react-icons/fa';
import { uiSelector } from 'features/ui/store/uiSelectors';
import IAIIconButton from 'common/components/IAIIconButton';
import { CloseIcon } from '@chakra-ui/icons';
import { useTranslation } from 'react-i18next';
import {
  floatingProgressImageMoved,
  floatingProgressImageResized,
  setShouldShowProgressImages,
} from 'features/ui/store/uiSlice';
import { Rnd } from 'react-rnd';
import { Rect } from 'features/ui/store/uiTypes';
import { isEqual } from 'lodash';

const selector = createSelector(
  [systemSelector, uiSelector],
  (system, ui) => {
    const { progressImage, isProcessing } = system;
    const {
      floatingProgressImageRect,
      shouldShowProgressImages,
      shouldAutoShowProgressImages,
    } = ui;

    const showProgressWindow =
      shouldAutoShowProgressImages && isProcessing
        ? true
        : shouldShowProgressImages;

    return {
      progressImage,
      floatingProgressImageRect,
      showProgressWindow,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

const ProgressImagePreview = () => {
  const dispatch = useAppDispatch();

  const { showProgressWindow, progressImage, floatingProgressImageRect } =
    useAppSelector(selector);

  const { t } = useTranslation();

  return showProgressWindow ? (
    <Rnd
      bounds="window"
      minHeight={200}
      minWidth={200}
      size={{
        width: floatingProgressImageRect.width,
        height: floatingProgressImageRect.height,
      }}
      position={{
        x: floatingProgressImageRect.x,
        y: floatingProgressImageRect.y,
      }}
      onDragStop={(e, d) => {
        dispatch(floatingProgressImageMoved({ x: d.x, y: d.y }));
      }}
      onResizeStop={(e, direction, ref, delta, position) => {
        const newRect: Partial<Rect> = {};

        console.log(ref.style.width, ref.style.height, position.x, position.y);

        if (ref.style.width) {
          newRect.width = ref.style.width;
        }
        if (ref.style.height) {
          newRect.height = ref.style.height;
        }
        if (position.x) {
          newRect.x = position.x;
        }
        if (position.x) {
          newRect.y = position.y;
        }

        dispatch(floatingProgressImageResized(newRect));
      }}
    >
      <Flex
        sx={{
          position: 'relative',
          w: 'full',
          h: 'full',
          alignItems: 'center',
          justifyContent: 'center',
          flexDir: 'column',
          boxShadow: 'dark-lg',
          bg: 'base.800',
          borderRadius: 'base',
        }}
      >
        <Flex
          sx={{
            w: 'full',
            alignItems: 'center',
            justifyContent: 'center',
            p: 1.5,
            pl: 4,
            pr: 3,
            bg: 'base.700',
            borderTopRadius: 'base',
          }}
        >
          <Flex
            sx={{
              flexGrow: 1,
              userSelect: 'none',
              cursor: 'move',
            }}
          >
            <Text fontSize="sm" fontWeight={500}>
              Progress Images
            </Text>
          </Flex>
          <IAIIconButton
            onClick={() => dispatch(setShouldShowProgressImages(false))}
            aria-label={t('ui.hideProgressImages')}
            size="xs"
            icon={<CloseIcon />}
            variant="ghost"
          />
        </Flex>
        {progressImage ? (
          <Flex
            sx={{
              position: 'relative',
              w: 'full',
              h: 'full',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Image
              draggable={false}
              src={progressImage.dataURL}
              width={progressImage.width}
              height={progressImage.height}
              sx={{
                position: 'absolute',
                objectFit: 'contain',
                maxWidth: '100%',
                maxHeight: '100%',
                height: 'auto',
                borderRadius: 'base',
                p: 2,
              }}
            />
          </Flex>
        ) : (
          <Flex
            sx={{
              w: 'full',
              h: 'full',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Icon color="base.400" boxSize={32} as={FaImage}></Icon>
          </Flex>
        )}
      </Flex>
    </Rnd>
  ) : (
    <IAIIconButton
      onClick={() => dispatch(setShouldShowProgressImages(true))}
      tooltip={t('ui.showProgressImages')}
      sx={{
        position: 'absolute',
        bottom: 4,
        insetInlineStart: 4,
      }}
      aria-label={t('ui.showProgressImages')}
      icon={<FaStopwatch />}
    />
  );
};

export default memo(ProgressImagePreview);
