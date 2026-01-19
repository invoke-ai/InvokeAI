import {
  Box,
  ButtonGroup,
  IconButton,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  Portal,
  Tooltip,
  useDisclosure,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useSelectTool, useToolIsSelected } from 'features/controlLayers/components/Tool/hooks';
import { selectGradientType, settingsGradientTypeChanged } from 'features/controlLayers/store/canvasSettingsSlice';
import { memo, useCallback, useEffect, useId } from 'react';
import { useTranslation } from 'react-i18next';

const GradientToolIcon = memo(() => {
  const id = useId();
  const gradientId = `${id}-gradient-tool-horizontal`;
  return (
    <Box as="svg" viewBox="0 0 24 24" boxSize={6} aria-hidden focusable={false} display="block">
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0.5" x2="1" y2="0.5">
          <stop offset="0%" stopColor="currentColor" stopOpacity="0.25" />
          <stop offset="100%" stopColor="currentColor" stopOpacity="0.85" />
        </linearGradient>
      </defs>
      <rect
        x="3"
        y="6"
        width="18"
        height="12"
        rx="2"
        fill={`url(#${gradientId})`}
        stroke="currentColor"
        strokeOpacity="0.9"
        strokeWidth="1"
      />
    </Box>
  );
});
GradientToolIcon.displayName = 'GradientToolIcon';

const GradientLinearIcon = memo(() => {
  const id = useId();
  const gradientId = `${id}-gradient-linear-diagonal`;
  return (
    <Box as="svg" viewBox="0 0 24 24" boxSize="22px" aria-hidden focusable={false} display="block">
      <defs>
        <linearGradient id={gradientId} x1="0" y1="1" x2="1" y2="0">
          <stop offset="0%" stopColor="currentColor" stopOpacity="0.25" />
          <stop offset="100%" stopColor="currentColor" stopOpacity="0.85" />
        </linearGradient>
      </defs>
      <rect
        x="4"
        y="4"
        width="16"
        height="16"
        rx="2"
        fill={`url(#${gradientId})`}
        stroke="currentColor"
        strokeOpacity="0.9"
        strokeWidth="1"
      />
    </Box>
  );
});
GradientLinearIcon.displayName = 'GradientLinearIcon';

const GradientRadialIcon = memo(() => {
  const id = useId();
  const gradientId = `${id}-gradient-radial`;
  return (
    <Box as="svg" viewBox="0 0 24 24" boxSize="22px" aria-hidden focusable={false} display="block">
      <defs>
        <radialGradient id={gradientId} cx="0.5" cy="0.5" r="0.5">
          <stop offset="0%" stopColor="currentColor" stopOpacity="0.25" />
          <stop offset="100%" stopColor="currentColor" stopOpacity="0.85" />
        </radialGradient>
      </defs>
      <circle
        cx="12"
        cy="12"
        r="8"
        fill={`url(#${gradientId})`}
        stroke="currentColor"
        strokeOpacity="0.9"
        strokeWidth="1"
      />
    </Box>
  );
});
GradientRadialIcon.displayName = 'GradientRadialIcon';

export const ToolGradientButton = memo(() => {
  const { t } = useTranslation();
  const isSelected = useToolIsSelected('gradient');
  const selectGradient = useSelectTool('gradient');
  const gradientType = useAppSelector(selectGradientType);
  const dispatch = useAppDispatch();
  const disclosure = useDisclosure();

  useEffect(() => {
    if (!isSelected) {
      disclosure.onClose();
    }
  }, [disclosure, isSelected]);

  const handleClick = useCallback(() => {
    selectGradient();
    if (disclosure.isOpen) {
      disclosure.onClose();
    } else {
      disclosure.onOpen();
    }
  }, [disclosure, selectGradient]);

  const setLinear = useCallback(() => {
    dispatch(settingsGradientTypeChanged('linear'));
  }, [dispatch]);

  const setRadial = useCallback(() => {
    dispatch(settingsGradientTypeChanged('radial'));
  }, [dispatch]);

  const gradientLabel = t('controlLayers.tool.gradient', { defaultValue: 'Gradient' });
  const linearLabel = t('controlLayers.gradient.linear', { defaultValue: t('common.linear', 'Linear') });
  const circularLabel = t('controlLayers.gradient.circular', { defaultValue: 'Circular' });

  return (
    <Popover
      isLazy
      isOpen={disclosure.isOpen}
      onClose={disclosure.onClose}
      closeOnBlur={true}
      closeOnEsc={true}
      placement="right-start"
      gutter={0}
    >
      <PopoverTrigger>
        <Tooltip label={gradientLabel} placement="end">
          <IconButton
            aria-label={gradientLabel}
            icon={<GradientToolIcon />}
            isActive={isSelected}
            colorScheme={isSelected ? 'invokeBlue' : 'base'}
            variant="solid"
            onClick={handleClick}
          />
        </Tooltip>
      </PopoverTrigger>
      <Portal>
        <PopoverContent width="auto" minW={0} ms={-20} mt={-7} bg="transparent" border="none" boxShadow="none">
          <PopoverArrow display="none" />
          <PopoverBody p={1}>
            <ButtonGroup isAttached size="sm" orientation="vertical">
              <Tooltip label={linearLabel}>
                <IconButton
                  aria-label={linearLabel}
                  icon={<GradientLinearIcon />}
                  colorScheme={gradientType === 'linear' ? 'invokeBlue' : 'base'}
                  variant="solid"
                  w="30px"
                  h="30px"
                  onClick={setLinear}
                />
              </Tooltip>
              <Tooltip label={circularLabel}>
                <IconButton
                  aria-label={circularLabel}
                  icon={<GradientRadialIcon />}
                  colorScheme={gradientType === 'radial' ? 'invokeBlue' : 'base'}
                  variant="solid"
                  w="30px"
                  h="30px"
                  onClick={setRadial}
                />
              </Tooltip>
            </ButtonGroup>
          </PopoverBody>
        </PopoverContent>
      </Portal>
    </Popover>
  );
});

ToolGradientButton.displayName = 'ToolGradientButton';
