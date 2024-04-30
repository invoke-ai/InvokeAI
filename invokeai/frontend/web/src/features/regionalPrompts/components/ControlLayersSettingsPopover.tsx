import { Flex, IconButton, Popover, PopoverBody, PopoverContent, PopoverTrigger } from '@invoke-ai/ui-library';
import { GlobalMaskLayerOpacity } from 'features/regionalPrompts/components/GlobalMaskLayerOpacity';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { RiSettings4Fill } from 'react-icons/ri';

const ControlLayersSettingsPopover = () => {
  const { t } = useTranslation();

  return (
    <Popover isLazy>
      <PopoverTrigger>
        <IconButton aria-label={t('common.settingsLabel')} icon={<RiSettings4Fill />} />
      </PopoverTrigger>
      <PopoverContent>
        <PopoverBody>
          <Flex direction="column" gap={2}>
            <GlobalMaskLayerOpacity />
          </Flex>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
};

export default memo(ControlLayersSettingsPopover);
