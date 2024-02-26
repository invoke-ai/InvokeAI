import { Box, Flex, FormControl, FormLabel, Icon, IconButton, Switch } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
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
import { PiCaretUpBold, PiCopyBold, PiTrashSimpleBold } from 'react-icons/pi';
import { useToggle } from 'react-use';

import ControlAdapterImagePreview from './ControlAdapterImagePreview';
import ControlAdapterProcessorComponent from './ControlAdapterProcessorComponent';
import ControlAdapterShouldAutoConfig from './ControlAdapterShouldAutoConfig';
import ControlNetCanvasImageImports from './imports/ControlNetCanvasImageImports';
import { ParamControlAdapterBeginEnd } from './parameters/ParamControlAdapterBeginEnd';
import ParamControlAdapterControlMode from './parameters/ParamControlAdapterControlMode';
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
    <Flex flexDir="column" gap={4} p={4} borderRadius="base" position="relative" bg="base.750">
      <Flex gap={2} alignItems="center" justifyContent="space-between">
        <FormControl>
          <FormLabel flexGrow={1}>{t(`controlnet.${controlAdapterType}`, { number })}</FormLabel>
          <Switch
            aria-label={t('controlnet.toggleControlNet')}
            isChecked={isEnabled}
            onChange={handleToggleIsEnabled}
          />
        </FormControl>
      </Flex>
      <Flex gap={4} alignItems="center">
        <Box minW={0} w="full" transitionProperty="common" transitionDuration="0.1s">
          <ParamControlAdapterModel id={id} />
        </Box>
        {activeTabName === 'unifiedCanvas' && <ControlNetCanvasImageImports id={id} />}
        <IconButton
          size="sm"
          tooltip={t('controlnet.duplicate')}
          aria-label={t('controlnet.duplicate')}
          onClick={handleDuplicate}
          icon={<PiCopyBold />}
        />
        <IconButton
          size="sm"
          tooltip={t('controlnet.delete')}
          aria-label={t('controlnet.delete')}
          colorScheme="error"
          onClick={handleDelete}
          icon={<PiTrashSimpleBold />}
        />
        <IconButton
          size="sm"
          tooltip={isExpanded ? t('controlnet.hideAdvanced') : t('controlnet.showAdvanced')}
          aria-label={isExpanded ? t('controlnet.hideAdvanced') : t('controlnet.showAdvanced')}
          onClick={toggleIsExpanded}
          variant="ghost"
          icon={
            <Icon
              boxSize={4}
              as={PiCaretUpBold}
              transform={isExpanded ? 'rotate(0deg)' : 'rotate(180deg)'}
              transitionProperty="common"
              transitionDuration="normal"
            />
          }
        />
      </Flex>

      <Flex w="full" flexDir="column" gap={4}>
        <Flex gap={8} w="full" alignItems="center">
          <Flex flexDir="column" gap={2} h={32} w="full">
            <ParamControlAdapterWeight id={id} />
            <ParamControlAdapterBeginEnd id={id} />
          </Flex>
          {!isExpanded && (
            <Flex alignItems="center" justifyContent="center" h={32} w={32} aspectRatio="1/1">
              <ControlAdapterImagePreview id={id} isSmall />
            </Flex>
          )}
        </Flex>
      </Flex>

      {isExpanded && (
        <>
          <Flex gap={2}>
            <ParamControlAdapterControlMode id={id} />
            <ParamControlAdapterResizeMode id={id} />
          </Flex>
          <ParamControlAdapterProcessorSelect id={id} />
          <ControlAdapterImagePreview id={id} />
          <ControlAdapterShouldAutoConfig id={id} />
          <ControlAdapterProcessorComponent id={id} />
        </>
      )}
    </Flex>
  );
};

export default memo(ControlAdapterConfig);
