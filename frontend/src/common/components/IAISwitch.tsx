import {
  Flex,
  FormControl,
  FormLabel,
  Switch,
  SwitchProps,
} from '@chakra-ui/react';
import { Feature } from '../../app/features';
import GuideIcon from '../../common/components/GuideIcon';

interface Props extends SwitchProps {
  label?: string;
  width?: string | number;
  feature?: Feature
}

/**
 * Customized Chakra FormControl + Switch multi-part component.
 */
const IAISwitch = (props: Props) => {
  const {
    label,
    isDisabled = false,
    fontSize = 'md',
    size = 'md',
    width = 'auto',
    feature,
    ...rest
  } = props;
  return (
    <FormControl isDisabled={isDisabled} width={width}>
      <Flex justifyContent={'space-between'} alignItems={'center'}>
        {label && (
          <FormLabel
            fontSize={fontSize}
            marginBottom={1}
            flexGrow={2}
            whiteSpace="nowrap"
          >
            {label}
          </FormLabel>
        )}
        <Switch size={size} className="switch-button" {...rest} />
        {feature && (
          <GuideIcon feature={feature} />
        )}
      </Flex>
    </FormControl>
  );
};

export default IAISwitch;
