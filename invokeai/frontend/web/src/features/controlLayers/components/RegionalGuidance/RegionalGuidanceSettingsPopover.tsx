import {
  Checkbox,
  Flex,
  FormControl,
  FormLabel,
  IconButton,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { stopPropagation } from 'common/util/stopPropagation';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { rgAutoNegativeChanged } from 'features/controlLayers/store/canvasV2Slice';
import { selectRegionalGuidanceEntityOrThrow } from 'features/controlLayers/store/regionsReducers';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiGearSixBold } from 'react-icons/pi';

export const RegionalGuidanceSettingsPopover = memo(() => {
  const entityIdentifier = useEntityIdentifierContext();
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const autoNegative = useAppSelector((s) => selectRegionalGuidanceEntityOrThrow(s.canvasV2, entityIdentifier.id).autoNegative);
  const onChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(rgAutoNegativeChanged({ id: entityIdentifier.id, autoNegative: e.target.checked ? 'invert' : 'off' }));
    },
    [dispatch, entityIdentifier.id]
  );

  return (
    <Popover isLazy>
      <PopoverTrigger>
        <IconButton
          tooltip={t('common.settingsLabel')}
          aria-label={t('common.settingsLabel')}
          size="sm"
          icon={<PiGearSixBold />}
          onDoubleClick={stopPropagation} // double click expands the layer
        />
      </PopoverTrigger>
      <PopoverContent>
        <PopoverArrow />
        <PopoverBody>
          <Flex direction="column" gap={2}>
            <FormControl gap={2}>
              <FormLabel flexGrow={1} minW={32} m={0}>
                {t('controlLayers.autoNegative')}
              </FormLabel>
              <Checkbox size="md" isChecked={autoNegative === 'invert'} onChange={onChange} />
            </FormControl>
          </Flex>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});

RegionalGuidanceSettingsPopover.displayName = 'RegionalGuidanceSettingsPopover';
