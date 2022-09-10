import {
  FormControl,
  FormLabel,
  HStack,
  Switch,
  SwitchProps,
} from '@chakra-ui/react';

interface Props extends SwitchProps {
  label: string;
  width?: string | number;
}

const SDSwitch = (props: Props) => {
  const {
    label,
    isDisabled = false,
    fontSize = 'sm',
    size = 'md',
    width,
    ...rest
  } = props;
  return (
    <FormControl isDisabled={isDisabled} width={width}>
      <HStack>
        <FormLabel
          marginInlineEnd={0}
          marginBottom={1}
          fontSize={fontSize}
          whiteSpace='nowrap'
        >
          {label}
        </FormLabel>
        <Switch size={size} {...rest} />
      </HStack>
    </FormControl>
  );
};

export default SDSwitch;
