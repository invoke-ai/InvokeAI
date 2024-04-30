import type { FormLabelProps } from '@invoke-ai/ui-library';
import {
  Flex,
  FormControlGroup,
  IconButton,
  Popover,
  PopoverArrow,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
} from '@invoke-ai/ui-library';
import { stopPropagation } from 'common/util/stopPropagation';
import { RGLayerAutoNegativeCheckbox } from 'features/controlLayers/components/RGLayerAutoNegativeCheckbox';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiGearSixBold } from 'react-icons/pi';

type Props = {
  layerId: string;
};

const formLabelProps: FormLabelProps = {
  flexGrow: 1,
  minW: 32,
};

const RGLayerSettingsPopover = ({ layerId }: Props) => {
  const { t } = useTranslation();

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
            <FormControlGroup formLabelProps={formLabelProps}>
              <RGLayerAutoNegativeCheckbox layerId={layerId} />
            </FormControlGroup>
          </Flex>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
};

export default memo(RGLayerSettingsPopover);
