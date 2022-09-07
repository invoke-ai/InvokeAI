import { FormControl, FormLabel, HStack, Switch } from '@chakra-ui/react';
import { ChangeEvent } from 'react';

type Props = {
  label: string;
  isChecked: boolean;
  onChange: (e: ChangeEvent<HTMLInputElement>) => void;
  isDisabled?: boolean;
};

const SDSwitch = ({
  label,
  isChecked,
  onChange,
  isDisabled = false,
}: Props) => {
  return (
    <FormControl isDisabled={isDisabled}>
      <HStack>
        <FormLabel
          marginInlineEnd={0}
          marginBottom={1}
          fontSize='sm'
          whiteSpace='nowrap'
        >
          {label}
        </FormLabel>
        <Switch onChange={onChange} isChecked={isChecked} />
      </HStack>
    </FormControl>
  );
};

export default SDSwitch;
