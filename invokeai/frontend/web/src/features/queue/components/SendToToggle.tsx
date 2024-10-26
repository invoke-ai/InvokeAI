import type { SystemStyleObject } from '@invoke-ai/ui-library';
import {
  Box,
  Button,
  chakra,
  Flex,
  Icon,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  Portal,
  Text,
  useCheckbox,
  useToken,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectSendToCanvas, settingsSendToCanvasChanged } from 'features/controlLayers/store/canvasSettingsSlice';
import { selectIsStaging } from 'features/controlLayers/store/canvasStagingAreaSlice';
import { useImageViewer } from 'features/gallery/components/ImageViewer/useImageViewer';
import { activeTabCanvasRightPanelChanged, setActiveTab } from 'features/ui/store/uiSlice';
import type { ChangeEvent, PropsWithChildren } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { Trans, useTranslation } from 'react-i18next';
import { PiImageBold, PiPaintBrushBold } from 'react-icons/pi';

const getSx = (padding: string | number): SystemStyleObject => ({
  bg: 'base.700',
  w: '72px',
  cursor: 'pointer',
  '&[data-checked]': {
    '.thumb': {
      left: `calc(100% - ${padding})`,
      transform: 'translateX(-100%)',
      bg: 'invokeGreen.300',
    },
    '.unchecked-icon': {
      color: 'base.50',
      opacity: 0.4,
    },
    '.checked-icon': {
      color: 'base.900',
      opacity: 1,
    },
  },
  '&[data-disabled]': {
    bg: 'base.700',
    '.thumb': {
      bg: 'base.500',
    },
    '.unchecked-icon': {
      color: 'base.800',
    },
    '.checked-icon': {
      color: 'base.800',
    },
  },
  '.thumb': {
    transition: 'left 0.1s ease-in-out, transform 0.1s ease-in-out',
    left: padding,
    transform: 'translateX(0)',
    bg: 'invokeBlue.400',
    shadow: 'md',
  },
  '.unchecked-icon': {
    color: 'base.900',
    opacity: 1,
  },
  '.checked-icon': {
    color: 'base.50',
    opacity: 0.4,
  },
});

export const SendToToggle = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const sendToCanvas = useAppSelector(selectSendToCanvas);
  const isStaging = useAppSelector(selectIsStaging);

  const gap = useToken('space', 1);
  const sx = useMemo(() => getSx(gap), [gap]);

  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(settingsSendToCanvasChanged(e.target.checked));
    },
    [dispatch]
  );

  const { getCheckboxProps, getInputProps, htmlProps } = useCheckbox({
    onChange,
    isChecked: sendToCanvas,
    isDisabled: isStaging,
  });

  return (
    <Popover trigger="hover">
      <PopoverTrigger>
        <chakra.label {...htmlProps}>
          <Flex
            position="relative"
            borderRadius="base"
            alignItems="center"
            justifyContent="space-between"
            h="full"
            p={gap}
            gap={gap}
            sx={sx}
            {...getCheckboxProps()}
          >
            <input {...getInputProps()} hidden />
            <Box className="thumb" position="absolute" borderRadius="base" w={10} top={gap} bottom={gap} />
            <Flex w={10} h="full" alignItems="center" justifyContent="center" pos="relative">
              <Icon
                className="unchecked-icon"
                w={6}
                h={6}
                as={PiImageBold}
                aria-label={t('controlLayers.sendToGallery')}
              />
            </Flex>
            <Flex w={10} h="full" alignItems="center" justifyContent="center" pos="relative">
              <Icon
                className="checked-icon"
                w={6}
                h={6}
                as={PiPaintBrushBold}
                aria-label={t('controlLayers.sendToCanvas')}
              />
            </Flex>
          </Flex>
        </chakra.label>
      </PopoverTrigger>
      <Portal>
        <PopoverContent maxW={296} p={2} bg="base.200" color="base.800">
          <PopoverArrow
            bg="base.200"
            left={sendToCanvas ? '18px !important' : '-18px !important'}
            transitionProperty="all"
            transitionDuration="0.2s"
          />
          <PopoverBody>
            <TooltipContent sendToCanvas={sendToCanvas} isStaging={isStaging} />
          </PopoverBody>
        </PopoverContent>
      </Portal>
    </Popover>
  );
});

SendToToggle.displayName = 'CanvasSendToToggle';

const TooltipContent = memo(({ sendToCanvas, isStaging }: { sendToCanvas: boolean; isStaging: boolean }) => {
  const { t } = useTranslation();

  if (isStaging) {
    return (
      <Flex flexDir="column">
        <Text fontWeight="semibold">{t('controlLayers.sendingToCanvas')}</Text>
        <Text fontWeight="normal">
          <Trans i18nKey="controlLayers.viewProgressOnCanvas" components={{ Btn: <ActivateCanvasButton /> }} />
        </Text>
      </Flex>
    );
  }

  return (
    <Flex flexDir="column">
      <Text fontWeight="semibold">
        {sendToCanvas ? t('controlLayers.sendToCanvas') : t('controlLayers.sendToGallery')}
      </Text>
      <Text fontWeight="normal">
        {sendToCanvas ? t('controlLayers.sendToCanvasDesc') : t('controlLayers.sendToGalleryDesc')}
      </Text>
    </Flex>
  );
});

TooltipContent.displayName = 'TooltipContent';

const ActivateCanvasButton = (props: PropsWithChildren) => {
  const dispatch = useAppDispatch();
  const imageViewer = useImageViewer();
  const onClick = useCallback(() => {
    dispatch(setActiveTab('canvas'));
    dispatch(activeTabCanvasRightPanelChanged('layers'));
    imageViewer.close();
  }, [dispatch, imageViewer]);
  return (
    <Button
      onClick={onClick}
      size="sm"
      variant="link"
      color="base.800"
      _hover={{ color: 'base.900', textDecoration: 'underline' }}
    >
      {props.children}
    </Button>
  );
};
