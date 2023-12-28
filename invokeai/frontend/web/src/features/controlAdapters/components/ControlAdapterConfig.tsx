import { ChevronUpIcon } from '@chakra-ui/icons';
import { Box, Flex } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import { InvSwitch } from 'common/components/InvSwitch/wrapper';
import { useControlAdapterIsEnabled } from 'features/controlAdapters/hooks/useControlAdapterIsEnabled';
import { useControlAdapterType } from 'features/controlAdapters/hooks/useControlAdapterType';
import {
  controlAdapterDuplicated,
  controlAdapterIsEnabledChanged,
  controlAdapterRemoved,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { FaCopy, FaTrash } from 'react-icons/fa';
import { useToggle } from 'react-use';

import ControlAdapterImagePreview from './ControlAdapterImagePreview';
import ControlAdapterProcessorComponent from './ControlAdapterProcessorComponent';
import ControlAdapterShouldAutoConfig from './ControlAdapterShouldAutoConfig';
import ControlNetCanvasImageImports from './imports/ControlNetCanvasImageImports';
import { ParamControlAdapterBeginEnd } from './parameters/ParamControlAdapterBeginEnd';
import ParamControlAdapterControlMode from './parameters/ParamControlAdapterControlMode';
import ParamControlAdapterModel from './parameters/ParamControlAdapterModel';
import ParamControlAdapterProcessorSelect from './parameters/ParamControlAdapterProcessorSelect';
import ParamControlAdapterResizeMode from './parameters/ParamControlAdapterResizeMode';
import ParamControlAdapterWeight from './parameters/ParamControlAdapterWeight';

const ControlAdapterConfig = (props: { id: string; number: number }) => {
  const { id, number } = props;
  const controlAdapterType = useControlAdapterType(id);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const activeTabName = useAppSelector(activeTabNameSelector);
  const isEnabled = useControlAdapterIsEnabled(id);
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
        bg: 'base.750',
      }}
    >
      <Flex
        sx={{ gap: 2, alignItems: 'center', justifyContent: 'space-between' }}
      >
        <InvControl label={t(`controlnet.${controlAdapterType}`, { number })}>
          <InvSwitch
            aria-label={t('controlnet.toggleControlNet')}
            isChecked={isEnabled}
            onChange={handleToggleIsEnabled}
          />
        </InvControl>
      </Flex>
      <Flex sx={{ gap: 2, alignItems: 'center' }}>
        <Box
          sx={{
            w: 'full',
            minW: 0,
            transitionProperty: 'common',
            transitionDuration: '0.1s',
          }}
        >
          <ParamControlAdapterModel id={id} />
        </Box>
        {activeTabName === 'unifiedCanvas' && (
          <ControlNetCanvasImageImports id={id} />
        )}
        <InvIconButton
          size="sm"
          tooltip={t('controlnet.duplicate')}
          aria-label={t('controlnet.duplicate')}
          onClick={handleDuplicate}
          icon={<FaCopy />}
        />
        <InvIconButton
          size="sm"
          tooltip={t('controlnet.delete')}
          aria-label={t('controlnet.delete')}
          colorScheme="error"
          onClick={handleDelete}
          icon={<FaTrash />}
        />
        <InvIconButton
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
                color: 'base.300',
                transform: isExpanded ? 'rotate(0deg)' : 'rotate(180deg)',
                transitionProperty: 'common',
                transitionDuration: 'normal',
              }}
            />
          }
        />
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
            <ParamControlAdapterWeight id={id} />
            <ParamControlAdapterBeginEnd id={id} />
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
              <ControlAdapterImagePreview id={id} isSmall />
            </Flex>
          )}
        </Flex>
        <Flex sx={{ gap: 2 }}>
          <ParamControlAdapterControlMode id={id} />
          <ParamControlAdapterResizeMode id={id} />
        </Flex>
        <ParamControlAdapterProcessorSelect id={id} />
      </Flex>

      {isExpanded && (
        <>
          <ControlAdapterImagePreview id={id} />
          <ControlAdapterShouldAutoConfig id={id} />
          <ControlAdapterProcessorComponent id={id} />
        </>
      )}
    </Flex>
  );
};

export default memo(ControlAdapterConfig);
