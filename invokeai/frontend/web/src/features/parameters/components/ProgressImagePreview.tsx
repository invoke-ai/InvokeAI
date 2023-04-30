import {
  Accordion,
  AccordionButton,
  AccordionIcon,
  AccordionItem,
  AccordionPanel,
  Box,
  Flex,
  Icon,
  Image,
  Text,
} from '@chakra-ui/react';
import { useDraggable } from '@dnd-kit/core';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { systemSelector } from 'features/system/store/systemSelectors';
import { memo, useCallback, useRef, MouseEvent } from 'react';
import { CSS } from '@dnd-kit/utilities';
import { FaEye, FaImage } from 'react-icons/fa';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { Resizable } from 're-resizable';
import { useBoolean, useHoverDirty } from 'react-use';
import IAIIconButton from 'common/components/IAIIconButton';
import { CloseIcon } from '@chakra-ui/icons';
import { useTranslation } from 'react-i18next';

const selector = createSelector([systemSelector, uiSelector], (system, ui) => {
  const { progressImage } = system;
  const { floatingProgressImageCoordinates, shouldShowProgressImage } = ui;

  return {
    progressImage,
    coords: floatingProgressImageCoordinates,
    shouldShowProgressImage,
  };
});

const ProgressImagePreview = () => {
  const { progressImage, coords, shouldShowProgressImage } =
    useAppSelector(selector);

  const [shouldShowProgressImages, toggleShouldShowProgressImages] =
    useBoolean(false);

  const { t } = useTranslation();
  const { attributes, listeners, setNodeRef, transform } = useDraggable({
    id: 'progress-image',
  });

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
            boundsByDirection={true}
            enable={{ bottomRight: true }}
          >
            <Flex
              sx={{
                cursor: 'move',
                w: 'full',
                h: 'full',
                alignItems: 'center',
                justifyContent: 'center',
              }}
              {...listeners}
              {...attributes}
            >
              <Flex
                sx={{
                  position: 'relative',
                  w: 'full',
                  h: 'full',
                  alignItems: 'center',
                  justifyContent: 'center',
                  p: 4,
                }}
              >
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
              <IAIIconButton
                onClick={toggleShouldShowProgressImages}
                aria-label={t('ui.hideProgressImages')}
                size="xs"
                icon={<CloseIcon />}
                sx={{
                  position: 'absolute',
                  top: 2,
                  insetInlineEnd: 2,
                }}
                variant="ghost"
              />
            </Flex>
          </Resizable>
        </Box>
      </Box>
    </Box>
  ) : (
    <IAIIconButton
      onClick={toggleShouldShowProgressImages}
      tooltip={t('ui.showProgressImages')}
      sx={{ position: 'absolute', bottom: 4, insetInlineStart: 4 }}
      aria-label={t('ui.showProgressImages')}
      icon={<FaEye />}
    />
  );
};

export default memo(ProgressImagePreview);
