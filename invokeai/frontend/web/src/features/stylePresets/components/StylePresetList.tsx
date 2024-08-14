import { Button, Collapse, Flex, Icon, Text, useDisclosure } from '@invoke-ai/ui-library';
import { PiCaretDownBold } from 'react-icons/pi';
import type { StylePresetRecordWithImage } from 'services/api/endpoints/stylePresets';

import { StylePresetListItem } from './StylePresetListItem';

export const StylePresetList = ({ title, data }: { title: string; data: StylePresetRecordWithImage[] }) => {
  const { onToggle, isOpen } = useDisclosure({ defaultIsOpen: true });

  if (!data.length) {
    return <></>;
  }

  return (
    <Flex flexDir="column">
      <Button variant="unstyled" onClick={onToggle}>
        <Flex gap={2} alignItems="center">
          <Icon boxSize={4} as={PiCaretDownBold} transform={isOpen ? undefined : 'rotate(-90deg)'} fill="base.500" />
          <Text fontSize="sm" fontWeight="semibold" userSelect="none" color="base.500">
            {title}
          </Text>
        </Flex>
      </Button>
      <Collapse in={isOpen}>
        {data.map((preset) => (
          <StylePresetListItem preset={preset} key={preset.id} />
        ))}
      </Collapse>
    </Flex>
  );
};
