import { Flex, Text, useDisclosure } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';
import IAICollapse from 'common/components/IAICollapse';
import { memo, useCallback, useState } from 'react';
import IAICustomSelect from 'common/components/IAICustomSelect';
import IAIIconButton from 'common/components/IAIIconButton';
import { FaPlus } from 'react-icons/fa';
import CannyProcessor from 'features/controlNet/components/processors/CannyProcessor';
import ControlNet from 'features/controlNet/components/ControlNet';

const CONTROLNET_MODELS = [
  'lllyasviel/sd-controlnet-canny',
  'lllyasviel/sd-controlnet-depth',
  'lllyasviel/sd-controlnet-hed',
  'lllyasviel/sd-controlnet-seg',
  'lllyasviel/sd-controlnet-openpose',
  'lllyasviel/sd-controlnet-scribble',
  'lllyasviel/sd-controlnet-normal',
  'lllyasviel/sd-controlnet-mlsd',
];

const ParamControlNetCollapse = () => {
  const { t } = useTranslation();
  const { isOpen, onToggle } = useDisclosure();
  const [model, setModel] = useState<string>(CONTROLNET_MODELS[0]);

  const handleSetControlNet = useCallback(
    (model: string | null | undefined) => {
      if (model) {
        setModel(model);
      }
    },
    []
  );

  return (
    <ControlNet />
    // <IAICollapse
    //   label={'ControlNet'}
    //   // label={t('parameters.seamCorrectionHeader')}
    //   isOpen={isOpen}
    //   onToggle={onToggle}
    // >
    //   <Flex sx={{ alignItems: 'flex-end' }}>
    //     <IAICustomSelect
    //       label="ControlNet Model"
    //       items={CONTROLNET_MODELS}
    //       selectedItem={model}
    //       setSelectedItem={handleSetControlNet}
    //     />
    //     <IAIIconButton
    //       size="sm"
    //       aria-label="Add ControlNet"
    //       icon={<FaPlus />}
    //     />
    //   </Flex>
    //   <CannyProcessor />
    // </IAICollapse>
  );
};

export default memo(ParamControlNetCollapse);
