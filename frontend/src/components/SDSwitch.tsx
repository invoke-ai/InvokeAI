import { FormControl, FormLabel, HStack, Switch } from '@chakra-ui/react';
import React, { ChangeEvent } from 'react';

type Props = {
  label: string;
  isChecked: boolean;
  onChange: (e: ChangeEvent<HTMLInputElement>) => void;
};

const SDSwitch = ({ label, isChecked, onChange }: Props) => {
  return (
    <FormControl>
      <HStack>
        <Switch size='md' onChange={onChange} isChecked={isChecked} />
        <FormLabel fontSize='md' whiteSpace='nowrap'>
          {label}
        </FormLabel>
      </HStack>
    </FormControl>
  );
};

export default SDSwitch;
