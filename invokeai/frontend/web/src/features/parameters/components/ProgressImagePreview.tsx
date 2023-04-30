import { Box, Flex, Icon, Image, Text } from '@chakra-ui/react';
import { useDraggable } from '@dnd-kit/core';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { systemSelector } from 'features/system/store/systemSelectors';
import { memo, useCallback } from 'react';
import { CSS } from '@dnd-kit/utilities';
import { FaEye, FaImage, FaStopwatch } from 'react-icons/fa';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { Resizable } from 're-resizable';
import IAIIconButton from 'common/components/IAIIconButton';
import { CloseIcon } from '@chakra-ui/icons';
import { useTranslation } from 'react-i18next';
import { shouldShowProgressImagesToggled } from 'features/ui/store/uiSlice';

const selector = createSelector([systemSelector, uiSelector], (system, ui) => {
  const { progressImage } = system;
  const { floatingProgressImageCoordinates, shouldShowProgressImages } = ui;

  return {
    progressImage,
    coords: floatingProgressImageCoordinates,
    shouldShowProgressImages,
  };
});

const ProgressImagePreview = () => {
  const dispatch = useAppDispatch();

  const { progressImage, coords, shouldShowProgressImages } =
    useAppSelector(selector);

  const { t } = useTranslation();

  const { attributes, listeners, setNodeRef, transform } = useDraggable({
    id: 'progress-image',
  });

  const toggleProgressImages = useCallback(() => {
    dispatch(shouldShowProgressImagesToggled());
  }, [dispatch]);

  const transformStyles = transform
    ? {
        transform: CSS.Translate.toString(transform),
      }
    : {};

  return shouldShowProgressImages ? (
    <Box
      sx={{
        position: 'absolute',
        left: `${coords.x}px`,
        top: `${coords.y}px`,
      }}
    >
      <Box ref={setNodeRef} sx={transformStyles}>
        <Box
          sx={{
            boxShadow: 'dark-lg',
            w: 'full',
            h: 'full',
            bg: 'base.800',
            borderRadius: 'base',
          }}
        >
          <Resizable
            defaultSize={{ width: 300, height: 300 }}
            minWidth={200}
            minHeight={200}
          >
            <Flex
              sx={{
                position: 'relative',
                w: 'full',
                h: 'full',
                alignItems: 'center',
                justifyContent: 'center',
                flexDir: 'column',
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
                  {...listeners}
                  {...attributes}
                >
                  <Text fontSize="sm" fontWeight={500}>
                    Progress Images
                  </Text>
                </Flex>
                <IAIIconButton
                  onClick={toggleProgressImages}
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
                      imageRendering: 'pixelated',
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
          </Resizable>
        </Box>
      </Box>
    </Box>
  ) : (
    <IAIIconButton
      onClick={toggleProgressImages}
      tooltip={t('ui.showProgressImages')}
      sx={{ position: 'absolute', bottom: 4, insetInlineStart: 4 }}
      aria-label={t('ui.showProgressImages')}
      icon={<FaStopwatch />}
    />
  );
};

export default memo(ProgressImagePreview);
