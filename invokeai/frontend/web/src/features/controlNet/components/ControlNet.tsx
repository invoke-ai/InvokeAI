import { Box, Flex } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { ChangeEvent, memo, useCallback } from 'react';
import { FaCopy, FaTrash } from 'react-icons/fa';
import {
  controlAdapterDuplicated,
  controlAdapterIsEnabledChanged,
  controlAdapterRemoved,
  selectControlAdapterById,
} from '../store/controlAdaptersSlice';
import ParamControlNetModel from './parameters/ParamControlNetModel';
import ParamControlNetWeight from './parameters/ParamControlNetWeight';

import { ChevronUpIcon } from '@chakra-ui/icons';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIIconButton from 'common/components/IAIIconButton';
import IAISwitch from 'common/components/IAISwitch';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { useTranslation } from 'react-i18next';
import { isControlNetOrT2IAdapter } from '../store/types';
import ControlNetImagePreview from './ControlNetImagePreview';
import ControlNetProcessorComponent from './ControlNetProcessorComponent';
import ParamControlNetShouldAutoConfig from './ParamControlNetShouldAutoConfig';
import ControlNetCanvasImageImports from './imports/ControlNetCanvasImageImports';
import ParamControlNetBeginEnd from './parameters/ParamControlNetBeginEnd';
import ParamControlNetControlMode from './parameters/ParamControlNetControlMode';
import ParamControlNetProcessorSelect from './parameters/ParamControlNetProcessorSelect';
import ParamControlNetResizeMode from './parameters/ParamControlNetResizeMode';
import { useToggle } from 'react-use';
import { useControlAdapterType } from '../hooks/useControlAdapterType';

const ControlNet = (props: { id: string }) => {
  const { id } = props;
  const controlAdapterType = useControlAdapterType(id);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const activeTabName = useAppSelector(activeTabNameSelector);

  const selector = createSelector(
    stateSelector,
    ({ controlAdapters }) => {
      const cn = selectControlAdapterById(controlAdapters, id);

      if (!cn) {
        return {
          isEnabled: false,
          shouldAutoConfig: false,
        };
      }

      const isEnabled = cn.isEnabled;
      const shouldAutoConfig = isControlNetOrT2IAdapter(cn)
        ? cn.shouldAutoConfig
        : false;

      return {
        isEnabled,
        shouldAutoConfig,
      };
    },
    defaultSelectorOptions
  );

  const { isEnabled, shouldAutoConfig } = useAppSelector(selector);
  const [isExpanded, toggleIsExpanded] = useToggle(false);

  const handleDelete = useCallback(() => {
    dispatch(controlAdapterRemoved({ id }));
  }, [id, dispatch]);

  const handleDuplicate = useCallback(() => {
    dispatch(controlAdapterDuplicated(id));
  }, [id, dispatch]);

  const handleToggleIsEnabled = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(
        controlAdapterIsEnabledChanged({
          id,
          isEnabled: e.target.checked,
        })
      );
    },
    [id, dispatch]
  );

  if (!controlAdapterType) {
    return null;
  }

  return (
    <Flex
      sx={{
        flexDir: 'column',
        gap: 3,
        p: 2,
        borderRadius: 'base',
        position: 'relative',
        bg: 'base.250',
        _dark: {
          bg: 'base.750',
        },
      }}
    >
      <Flex sx={{ gap: 2, alignItems: 'center' }}>
        <IAISwitch
          tooltip={t('controlnet.toggleControlNet')}
          aria-label={t('controlnet.toggleControlNet')}
          isChecked={isEnabled}
          onChange={handleToggleIsEnabled}
        />
        <Box
          sx={{
            w: 'full',
            minW: 0,
            // opacity: isEnabled ? 1 : 0.5,
            // pointerEvents: isEnabled ? 'auto' : 'none',
            transitionProperty: 'common',
            transitionDuration: '0.1s',
          }}
        >
          <ParamControlNetModel id={id} />
        </Box>
        {activeTabName === 'unifiedCanvas' && (
          <ControlNetCanvasImageImports id={id} />
        )}
        <IAIIconButton
          size="sm"
          tooltip={t('controlnet.duplicate')}
          aria-label={t('controlnet.duplicate')}
          onClick={handleDuplicate}
          icon={<FaCopy />}
        />
        <IAIIconButton
          size="sm"
          tooltip={t('controlnet.delete')}
          aria-label={t('controlnet.delete')}
          colorScheme="error"
          onClick={handleDelete}
          icon={<FaTrash />}
        />
        <IAIIconButton
          size="sm"
          tooltip={
            isExpanded
              ? t('controlnet.hideAdvanced')
              : t('controlnet.showAdvanced')
          }
          aria-label={
            isExpanded
              ? t('controlnet.hideAdvanced')
              : t('controlnet.showAdvanced')
          }
          onClick={toggleIsExpanded}
          variant="ghost"
          sx={{
            _hover: {
              bg: 'none',
            },
          }}
          icon={
            <ChevronUpIcon
              sx={{
                boxSize: 4,
                color: 'base.700',
                transform: isExpanded ? 'rotate(0deg)' : 'rotate(180deg)',
                transitionProperty: 'common',
                transitionDuration: 'normal',
                _dark: {
                  color: 'base.300',
                },
              }}
            />
          }
        />

        {!shouldAutoConfig && (
          <Box
            sx={{
              position: 'absolute',
              w: 1.5,
              h: 1.5,
              borderRadius: 'full',
              top: 4,
              insetInlineEnd: 4,
              bg: 'accent.700',
              _dark: {
                bg: 'accent.400',
              },
            }}
          />
        )}
      </Flex>

      <Flex sx={{ w: 'full', flexDirection: 'column', gap: 3 }}>
        <Flex sx={{ gap: 4, w: 'full', alignItems: 'center' }}>
          <Flex
            sx={{
              flexDir: 'column',
              gap: 3,
              h: 28,
              w: 'full',
              paddingInlineStart: 1,
              paddingInlineEnd: isExpanded ? 1 : 0,
              pb: 2,
              justifyContent: 'space-between',
            }}
          >
            <ParamControlNetWeight id={id} />
            <ParamControlNetBeginEnd id={id} />
          </Flex>
          {!isExpanded && (
            <Flex
              sx={{
                alignItems: 'center',
                justifyContent: 'center',
                h: 28,
                w: 28,
                aspectRatio: '1/1',
              }}
            >
              <ControlNetImagePreview id={id} isSmall />
            </Flex>
          )}
        </Flex>
        <Flex sx={{ gap: 2 }}>
          <ParamControlNetControlMode id={id} />
          <ParamControlNetResizeMode id={id} />
        </Flex>
        <ParamControlNetProcessorSelect id={id} />
      </Flex>

      {isExpanded && (
        <>
          <ControlNetImagePreview id={id} />
          <ParamControlNetShouldAutoConfig id={id} />
          <ControlNetProcessorComponent id={id} />
        </>
      )}
    </Flex>
  );
};

export default memo(ControlNet);
