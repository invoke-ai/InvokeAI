import { Box, Flex } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { ChangeEvent, memo, useCallback } from 'react';
import { FaCopy, FaTrash } from 'react-icons/fa';
import {
  controlAdapterDuplicated,
  controlAdapterIsEnabledChanged,
  controlAdapterRemoved,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import ParamControlAdapterModel from './parameters/ParamControlAdapterModel';
import ParamControlAdapterWeight from './parameters/ParamControlAdapterWeight';

import { ChevronUpIcon } from '@chakra-ui/icons';
import IAIIconButton from 'common/components/IAIIconButton';
import IAISwitch from 'common/components/IAISwitch';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { useTranslation } from 'react-i18next';
import { useToggle } from 'react-use';
import { useControlAdapterIsEnabled } from 'features/controlAdapters/hooks/useControlAdapterIsEnabled';
import { useControlAdapterType } from 'features/controlAdapters/hooks/useControlAdapterType';
import ControlAdapterImagePreview from './ControlAdapterImagePreview';
import ControlAdapterProcessorComponent from './ControlAdapterProcessorComponent';
import ControlAdapterShouldAutoConfig from './ControlAdapterShouldAutoConfig';
import ControlNetCanvasImageImports from './imports/ControlNetCanvasImageImports';
import ParamControlAdapterBeginEnd from './parameters/ParamControlAdapterBeginEnd';
import ParamControlAdapterControlMode from './parameters/ParamControlAdapterControlMode';
import ParamControlAdapterProcessorSelect from './parameters/ParamControlAdapterProcessorSelect';
import ParamControlAdapterResizeMode from './parameters/ParamControlAdapterResizeMode';

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
        bg: 'base.250',
        _dark: {
          bg: 'base.750',
        },
      }}
    >
      <Flex
        sx={{ gap: 2, alignItems: 'center', justifyContent: 'space-between' }}
      >
        <IAISwitch
          label={t(`controlnet.${controlAdapterType}`, { number })}
          aria-label={t('controlnet.toggleControlNet')}
          isChecked={isEnabled}
          onChange={handleToggleIsEnabled}
          formControlProps={{ w: 'full' }}
          formLabelProps={{ fontWeight: 600 }}
        />
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
